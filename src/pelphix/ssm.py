# !/usr/bin/env python3
import logging
import math
import multiprocessing as mp
import os
import sys
sys.path.append('/data1/sampath/prephixSynth/src')
from prephix.data.handlers.utils import segment_ct_3, segment_ct_total_mr
from pathlib import Path

from typing import Optional, List, Any, Tuple
from hydra.utils import get_original_cwd
import numpy as np
from skimage.transform import resize
import pyvista as pv
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from rich.progress import track

# Nosort
from .utils.onedrive_utils import OneDrive
from .shapes.ssm import StatisticalShapeModel
from .shapes.mesh import Mesh
from deepdrr.utils import data_utils

log = logging.getLogger(__name__)

pelvis_n_points = 10000
n_points = 5000

def pelvis_ssm_build(cfg : DictConfig):
    log.info("----------------- Propagating hip annotations -----------------")

    # Load SSM annotations
    pelvis_ssm_annotation_dir = Path('/data1/sampath/pelphix/data/annotations_pelvis_ssm_0337-meshes_010k-pts_40-comps')
    if not pelvis_ssm_annotation_dir.exists():
        log.info(f"No annotations found at {pelvis_ssm_annotation_dir}")
        exit()
    pelvis_ssm_path = Path('/data1/sampath/pelphix/data/pelvis_ssm_0337-meshes_010k-pts_40-comps.npz')
    pelvis_ssm = StatisticalShapeModel.from_npz(pelvis_ssm_path)
    pelvis_ssm.read_annotations(pelvis_ssm_annotation_dir)
    log.debug(f"Loaded {len(pelvis_ssm.annotations)} annotations from {pelvis_ssm_annotation_dir}")

    # Create output directory for propagated annotations
    pelvis_annotations_dir: Path = Path("/data1/sampath/SynthPelvisAnnotations")
    if not pelvis_annotations_dir.exists():
        pelvis_annotations_dir.mkdir(parents=True)
        
    mesh_dir = Path("/data1/sampath/TotalSegmentator_mesh/cad_rep_id_csv_sr")
    nifti_dir = Path('/data1/bohua/NMDID-ARCADE/nifti/cad_rep_id_csv_sr')

    # Outputs
    pelvis_mesh_paths = []  # List of [hip_left_path, hip_right_path, sacrum_path]
    pelvis_case_names = []  # List of case directory names

    # Iterate through each case directory
    #for case_dir in mesh_dir.iterdir():
    #    if not case_dir.is_dir():
    #        continue  # Skip non-directory files
#
    #    # Paths to the required mesh files
    #    hip_left_path = case_dir / "hip_left.stl"
    #    hip_right_path = case_dir / "hip_right.stl"
    #    sacrum_path = case_dir / "sacrum.stl"
#
    #    # Ensure all required meshes exist
    #    if not hip_left_path.exists() or not hip_right_path.exists() or not sacrum_path.exists():
    #        log.error(f"Missing one or more required meshes in {case_dir} (skipping)")
    #        continue
#
    #    # Append valid paths to the list
    #    pelvis_mesh_paths.append([hip_left_path, hip_right_path, sacrum_path])
    #    pelvis_case_names.append(case_dir.name)
    
    for case_dir in nifti_dir.iterdir():
        if not case_dir.is_dir():
            continue  # Skip non-directory files

        case_name = case_dir.name
        ct_path = case_dir / f"{case_name}.nii.gz"  # Adjust filename as needed
        #log.info(ct_path)
        #log.info(case_dir)
        #log.info(case_name)
        # Check if the case_name exists in mesh_dir
        mesh_case_dir = mesh_dir / case_name
        if mesh_case_dir.exists() and mesh_case_dir.is_dir():
            # Paths to the required mesh files
            hip_left_path = mesh_case_dir / "hip_left.stl"
            hip_right_path = mesh_case_dir / "hip_right.stl"
            sacrum_path = mesh_case_dir / "sacrum.stl"

            # Ensure all required meshes exist
            if hip_left_path.exists() and hip_right_path.exists() and sacrum_path.exists():
                pelvis_mesh_paths.append([hip_left_path, hip_right_path, sacrum_path])
                pelvis_case_names.append(case_name)
            else:
                if ct_path.exists():
                    log.info(f"Case {case_name} meshes are missing. Segmenting...")
                    segment_ct_total_mr(ct_path = ct_path, case_id = case_name, root_dir = None)
        else:
            # Case not found in mesh_dir, call segment_ct_3
            if ct_path.exists():
                log.info(f"Case {case_name} not found in mesh_dir. Segmenting...")
                segment_ct_total_mr(ct_path = ct_path, case_id = case_name, root_dir = None)
            else:
                log.error(f"CT file not found for case {case_name} in {ct_path} (skipping)")

    log.info(f"Processed {len(pelvis_mesh_paths)} cases successfully.")
    # Iterate through all synthetic CT meshes
    for i, pelvis_paths in enumerate(track(pelvis_mesh_paths, description="Propagating landmarks")):
        log.info(f"Propagating landmarks for {pelvis_paths} ({i+1}/{len(pelvis_mesh_paths)})")

        # Create case-specific directory for annotations
        case_annotations_dir = pelvis_annotations_dir / pelvis_case_names[i]
        case_annotations_dir.mkdir(parents=True, exist_ok=True)

        # Check if all annotations for this case already exist
        case_annotations = set(p.stem for p in case_annotations_dir.glob("*.fcsv"))
        if all(annotation_name in case_annotations for annotation_name in pelvis_ssm.annotations):
            log.debug(f"Annotations already exist for: {case_annotations}")
            continue

        # Combine mesh files for the current case into a single mesh
        pv_mesh = pv.PolyData()
        for p in pelvis_paths:
            pv_mesh += pv.read(str(p))

        # Decimate the mesh to reduce complexity
        pv_mesh.decimate(1 - pelvis_n_points / pv_mesh.n_points, inplace=True)
        mesh = Mesh.from_pv(pv_mesh)
        points_target = np.array(mesh.vertices)

        # Check if weights exist for this case; load if available
        weights_path = case_annotations_dir / f"weights.npy"
        if weights_path.exists():
            log.info(f"Loading weights from {weights_path}")
            weights = np.load(weights_path)
            freeze_weights = True  # Freeze weights to avoid recomputation
        else:
            weights = None
            freeze_weights = False

        # Fit the SSM to the target mesh
        try:
            weights, deformed_model, _ = pelvis_ssm.fit(
                points_target, max_iterations=20, weights=weights, freeze_weights=freeze_weights
            )
        except RuntimeError:
            log.error(f"Failed to fit SSM to {pelvis_paths}")
            continue

        # Propagate annotations and calculate distances
        distances = deformed_model.project_annotations(mesh)
        log.info(f"Mean/max distance: {np.mean(distances):.2f}/{np.max(distances):.2f} mm")

        # Save propagated annotations, weights, and distances
        mesh.save_annotations(case_annotations_dir)
        np.save(weights_path, weights)
        np.save(case_annotations_dir / f"distances.npy", distances)

        # Clear mesh to save memory
        mesh = None

def hip_ssm_build(cfg: DictConfig):
    log.info("----------------- Propagating hip landmarks -----------------")

    # Load SSM annotations
    ssm_annotation_dir = Path('/data1/sampath/pelphix/data/annotations_hip_ssm_0674-meshes_005k-pts_40-comps')
    if not ssm_annotation_dir.exists():
        log.info(f"No annotations found at {ssm_annotation_dir}")
        exit()
    hip_ssm = StatisticalShapeModel.from_npz(Path('/data1/sampath/pelphix/data/hip_ssm_0674-meshes_005k-pts_40-comps.npz'))
    hip_ssm.read_annotations(ssm_annotation_dir)
    if not "R_landmarks" in hip_ssm.annotations:
        log.debug(f"Annotations: {hip_ssm.annotations.keys()}")
        log.info("No landmarks found in annotations")
        exit()

    # Set up output directory for annotations
    annotations_dir = Path("/data1/sampath/SynthHipAnnotations")
    if not annotations_dir.exists():
        annotations_dir.mkdir(parents=True)


    # Directory containing case subdirectories
    base_dir = Path("/data1/sampath/TotalSegmentator_mesh/cad_rep_id_csv_sr")

    # Outputs
    hip_paths: list[Path] = []  # Paths to hip_left.stl and hip_right.stl
    case_names: list[str] = []  # Case directory names
    cache_names: list[str] = []  # Unique cache names combining case name and side

    # Iterate through case directories
    for case_dir in track(base_dir.iterdir(), description="Getting hip paths"):
        if not case_dir.is_dir():
            continue  # Skip non-directory files

        # Check for both left and right hip files
        for side in ["left", "right"]:
            hip_path = case_dir / f"hip_{side}.stl"
            if not hip_path.exists():
                continue  # Skip if the file doesn't exist

            # Append valid paths and associated names
            hip_paths.append(hip_path)
            cache_names.append(f"{case_dir.name}_{side}")
            case_names.append(case_dir.name)

    log.info(f"Processed {len(hip_paths)} hip files.")

    # Iterate through all synthetic hip meshes
    for i, hip_path in enumerate(track(hip_paths, description="Propagating landmarks")):
        log.info(f"Propagating landmarks for {hip_path} ({i+1}/{len(hip_paths)})")

        # Determine if this is the right or left hip
        right_side = "right" in hip_path.name
        side = "right" if right_side else "left"

        # Create a case-specific directory for annotations
        case_annotations_dir = annotations_dir / case_names[i]
        case_annotations_dir.mkdir(parents=True, exist_ok=True)

        # Define paths for landmarks and weights
        landmarks_path = case_annotations_dir / "{}_landmarks.fcsv".format(
            "R" if right_side else "L"
        )
        weights_path = case_annotations_dir / f"weights_{side}.npy"

        # Skip cases where landmarks already exist
        if landmarks_path.exists():
            log.info(f"Landmarks already exist at {landmarks_path}")
            continue

        # Load and preprocess the hip mesh
        pv_mesh = pv.read(str(hip_path))
        pv_mesh.decimate(1 - n_points / pv_mesh.n_points, inplace=True)
        mesh = Mesh.from_pv(pv_mesh)

        # Flip left hip meshes to align with the right side
        if not right_side:
            mesh = mesh.fliplr()

        # Fit the SSM to the target mesh
        points_target = np.array(mesh.vertices)
        if weights_path.exists():
            # Load existing weights and freeze them
            weights = np.load(weights_path)
            weights, deformed_model, _ = hip_ssm.fit(
                points_target, max_iterations=20, weights=weights, fix_weights=True
            )
        else:
            # Fit without pre-existing weights
            weights, deformed_model, _ = hip_ssm.fit(points_target, max_iterations=20)

        # Project annotations onto the mesh
        distances = deformed_model.project_annotations(mesh)

        # Log projection accuracy
        log.info(f"Mean/max distance: {np.mean(distances):.2f}/{np.max(distances):.2f} mm")

        # Flip the mesh back for saving if it was flipped earlier
        if not right_side:
            mesh = mesh.fliplr()

        # Save annotations, weights, and distances
        mesh.save_annotations(case_annotations_dir)
        np.save(weights_path, weights)
        np.save(case_annotations_dir / f"distances.npy", distances)

        # Clear the mesh to save memory
        mesh = None
            
def ssm_build(cfg: DictConfig):
    pelvis_ssm_build(cfg)
    hip_ssm_build(cfg)
