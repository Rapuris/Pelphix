<div align="center">

# Pelphix

Surgical Phase Recognition from X-ray Images in Percutaneous Pelvic Fixation

![Pelphix](images/procedure_000.gif)

</div>

<div align="center">

<!-- TODO: update links to the arxiv and dataset links. -->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](
<https://opensource.org/licenses/Apache-2.0>)
[![arXiv](https://img.shields.io/badge/arXiv-2109.13900-b31b1b.svg)](
<https://arxiv.org/abs/2109.13900>)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5555555.svg)](
<https://doi.org/10.5281/zenodo.5555555>)

<div align="left">

## Overview

Surgical phase recognition (SPR) is a crucial element in the digital transformation of the modern operating theater. While SPR based on video sources is well-established, incorporation of interventional X-ray sequences has not yet been explored. This paper presents Pelphix, a first approach to SPR for X-ray-guided percutaneous pelvic fracture fixation, which models the procedure at four levels of granularity – corridor, activ- ity, view, and frame value – simulating the pelvic fracture fixation work- flow as a Markov process to provide fully annotated training data. Using added supervision from detection of bony corridors, tools, and anatomy, we learn image representations that are fed into a transformer model to regress surgical phases at the four granularity levels. Our approach demonstrates the feasibility of X-ray-based SPR, achieving an average accuracy of 93.8% on simulated sequences and 67.57% in cadaver across all granularity levels, with up to 88% accuracy for the target corridor in real data. This work constitutes the first step toward SPR for the X-ray domain, establishing an approach to categorizing phases in X-ray-guided surgery, simulating realistic image sequences to enable machine learning model development, and demonstrating that this approach is feasible for the analysis of real procedures. As X-ray-based SPR continues to ma- ture, it will benefit procedures in orthopedic surgery, angiography, and interventional radiology by equipping intelligent surgical systems with situational awareness in the operating room.

## Data

<!-- TODO: add download links when available. -->

The simulated training and validation data can be downloaded here.

| Download | Training Images | Val Images |  Download Size |
| ------------ | -------- | ------------ | ------------- |
| pelphix_000338 | 139,922 | 4,285 | 3.2 GB |
| pelphix_000339 | 139,787 | 4,230 | 3.2 GB |
| **Total** | **279,709** | **8,515** | **6.4 GB** |

Sequences from our cadaveric experiments are available from the following links:

| Download | Images | Download Size |
| ------------ | -------- | ------------- |
| liverpool | 256 | 1.2 GB |

## Installation

Clone the repository:

```bash
git clone git@github.com:benjamindkilleen/pelphix.git
git submodule init
git submodule update
```

Install and activate the conda environment with

```bash
conda env create -f environment.yaml
conda activate pelphix
```

## Usage

### Run Experiments

Individual experiments can be run by specifying the `experiment` argument to `main.py`. For example,

```bash
python main.py experiment={ssm,generate,pretrain,train,test} [options]
```

- `ssm` runs the statistical shape model to propagate annotations.
- `generate` generates simulated datasets for sequences and view-invariant (totally random) sampling.
- `pretrain` pre-trains the model on view-invariant data.
- `train` trains the model on simulated sequences.
- `test` tests the model on simulated sequences and cadaver data.

### Options

See [conf/config.yaml](/conf/config.yaml) for a full list of options. Common variations are:

- `ckpt=/path/to/last.ckpt` to load a model checkpoint for resuming training or running inference.
- `gpus=n` to use `n` GPUs.

## Notes

### Model Checkpoints

- Dumbledore: full model
- Sirius: full model, same as dumbledor, but with scheduler, and no class weighting. (keypoints, segs, etc.)
- dobby: keypoints=False, segs=True
- on pacman: hagrid (not yet named) (keypoints=False, segs=False)

### TODO

- Ablation study on model architecture.
- Visualization of sim-to-real transfer. (e.g. register real images, get GT annotations, show drop in overall performance)
- Discussion of training/validation split.

## Citation

If you found this work useful, please cite [our paper](https://arxiv.org/abs/2304.09285):

```bibtex
@article{Pelphix2023,
 author = {Killeen, Benjamin D. and Zhang, Han and Mangulabnan, Jan and Armand, Mehran and Taylor, Russel H. and Osgood, Greg and Unberath, Mathias},
 title = {{Pelphix: Surgical Phase Recognition from X-ray Images in Percutaneous Pelvic Fixation}},
 journal = {arXiv},
 year = {2023},
 month = apr,
 eprint = {2304.09285},
 doi = {10.48550/arXiv.2304.09285}
}
```

If you use the simulated data, please also cite the NMDID database:

```bibtex
@misc{NMDID2020,
  author = {Edgar, HJH and Daneshvari Berry, S and Moes, E and Adolphi, NL and Bridges, P and Nolte, KB},
  title = {New Mexico Decedent Image Database},
  year = {2020},
  howpublished = {Office of the Medical Investigator, University of New Mexico},
  doi = {10.25827/5s8c-n515},
}
```

</div>
