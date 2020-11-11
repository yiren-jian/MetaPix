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
To train MetaPixelWeight:
```bash
$ cd <root_dir>/advent/scripts
$ python train.py --cfg ./configs/meta_pixel_weight.yml
```

### Testing
To test MetaPixelWeight:
```bash
$ cd <root_dir>/advent/scripts
$ python test.py --cfg ./configs/meta_pixel_weight.yml
```

## Acknowledgements
This codebase is heavily borrowed from [ADVENT](https://github.com/valeoai/ADVENT).
