# META DOMAIN TRANSFER
This repo covers the implementation of **[MetaPix: Domain transfer for semantic segmentation by meta pixel weighting](https://www.sciencedirect.com/science/article/pii/S0262885621002390)** published on Image and Vision Computing, 2021.
```bibtex
@article{Jian2021MetaPix,
author = {Yiren Jian and Chongyang Gao},
title = {MetaPix: Domain Transfer for Semantic Segmentation by Meta Pixel Weighting},
journal = {Image and Vision Computing},
pages = {104334},
year = {2021},
issn = {0262-8856}
}
```

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 1.5
* CUDA 9.2 or higher

### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/yiren-jian/META_DOMAIN_TRANSFER
$ cd META_DOMAIN_TRANSFER
```

1. Install this repository and the dependencies using pip:
```bash
$ pip install -e <root_dir>
```

2. Optional. To uninstall this package, run:
```bash
$ pip uninstall META_DOMAIN_TRANSFER
```

## Running the code
### Preparation
```bash
$ cd <root_dir>/meta_domain_transfer/model
$ python fcn8s_v2.py
```
This will create a directory `pretrained_models` with/and a FCN model pretrained on ImageNet.

### Training
To train MetaPixelWeight:

```bash
$ cd <root_dir>/meta_domain_transfer/scripts
$ python train.py --cfg ./configs/direct_joint.yml
```

```bash
$ cd <root_dir>/meta_domain_transfer/scripts
$ python train.py --cfg ./configs/train_W_with_FCN.yml
```

```bash
$ cd <root_dir>/meta_domain_transfer/scripts
$ python train.py --cfg ./configs/train_FCN_with_W.yml
```


### Testing
To test MetaPixelWeight:
```bash
$ cd <root_dir>/meta_domain_transfer/scripts
$ python test.py --cfg ./configs/train_FCN_with_W.yml
```

### Baselines

```bash
$ cd <root_dir>/meta_domain_transfer/scripts
$ python train.py --cfg ./configs/direct_joint.yml
$ python test.py --cfg ./configs/direct_joint.yml
```

```bash
$ cd <root_dir>/meta_domain_transfer/scripts
$ python train.py --cfg ./configs/target_only.yml
$ python test.py --cfg ./configs/target_only.yml
```

## Acknowledgements
This codebase is heavily borrowed from [ADVENT](https://github.com/valeoai/ADVENT).
