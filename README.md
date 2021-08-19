# META DOMAIN TRANSFER
GTA5/Synthia to Cityscapes

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
