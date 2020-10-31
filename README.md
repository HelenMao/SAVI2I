[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# SAVI2I: Continuous and Diverse Image-to-Image Translation via Signed Attribute Vectors

<img src='imgs/teaser.png' width="1200px">

Pytorch implementation for SAVI2I. We propose a simple yet effective signed attribute vector (SAV) that facilitates **continuous** translation on **diverse** mapping paths across **multiple** domains. 
<br>
More video results please see [Our Webpage](https://helenmao.github.io/SAVI2I/)
<br>
Contact: Qi Mao (qimao@pku.edu.cn)

## Paper
Continuous and Diverse Image-to-Image Translation via Signed Attribute Vectors<br>
[Qi Mao](https://sites.google.com/view/qi-mao/), [Hsin-Ying Lee](https://research.snap.com/team/hsin-ying-lee/), [Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), [Siwei Ma](https://scholar.google.com/citations?user=y3YqlaUAAAAJ&hl=zh-CN), and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)<br>
In arXiv 2020

## Citation
If you find this work useful for your research, please cite our paper:

```

```

## Quick Start

### Prerequisites
- Python 3.6
- Pytorch 0.4.0+

### Hardware Requirement
Suggest to use two P100 16GB GPUs or One V100 32GB GPU.


### Install
- Clone this repo:
```
git clone https://github.com/HelenMao/SAVI2I.git
```
### Training Datasets
Download datasets for each task into the dataset folder
```
mkdir datasets
```
- Yosemite  (summer <-> winter) 
- Photo2Artworks (Photo, Monet, Van Gogh and Ukiyo-e) <br>
You can follow the instructions of [CycleGAN datasets](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md) to download Yosemite and Photo2Artworks datasets.

- CelebA-HQ (Male  <-> Female)  <br> 
We split CelebA-HQ into male and female domains according to the annotated label and fine-tune the images manaully. 
- [AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq)  (Cat, Dog and WildLife)

## Training

### Notes
> For low-level style translation tasks, you suggest to set ```--type=1``` to use corresponding network architectures. <br>
> For shape-variation translation tasks, you suggest to set ```--type=0``` to use corresponding network architectures.


- Yosemite
```
python train.py --dataroot ./datasets/Yosemite/ --phase train --type 1 --name yosemite --n_ep 700 --n_ep_decay 500 --lambda_r1 10 --lambda_mmd 1 --num_domains 2
```
- Photo2Artwork
```
python train.py --dataroot ./datasets/Photo2Artworks/ --phase train --type 1 --name photo2artworks --n_ep 100 --n_ep_decay 0 --lambda_r1 10 --lambda_mmd 1 --num_domains 4
```
- CelebAHQ
```
python train.py --dataroot ./datasets/CelebAHQ/ --phase train --type 0 --name celebAHQ --n_ep 30 --n_ep_decay 0 --lambda_r1 1 --lambda_mmd 1 --num_domains 2
```
- AFHQ
```
python train.py --dataroot ./datasets/AFHQ/ --phase train --type 0 --name AFHQ --n_ep 100 --n_ep_decay 0 --lambda_r1 1 --lambda_mmd 10 --num_domains 3
```


## Pre-trained Models
- [Yosemite](https://drive.google.com/file/d/1relOFLfOW0ACpr_u6DXgll7Qf6sLSstx/view?usp=sharing) 
- [Photo2Artwork](https://drive.google.com/file/d/1B1G_Ml-a0phvBG_ePlbwBOpT4hOk9X9h/view?usp=sharing)
- [CelebAHQ](https://drive.google.com/file/d/1x0sRX-QTQ3z5Eep-ROmX9wcAMHWIBT6j/view?usp=sharing)
- [AFHQ](https://drive.google.com/open?id=1tnDDolN-OMLG4BUNB6rPIjSXoP2FbXgw)

Download and save them into 
```
./models/
```

## Testing 
**Reference-guided**
```
python test_reference_save.py --dataroot ./datasets/CelebAHQ --resume ./models/CelebAHQ/00029.pth --phase test --type 0 --num_domains 2 --index_s A --index_t B --num 5 --name CelebAHQ_ref  
```
**Latent-guided** 
```
python test_latent_rdm_save.py --dataroot ./datasets/CelebAHQ --resume ./models/CelebAHQ/00029.pth --phase test --type 0 --num_domains 2 --index_s A --index_t B --num 5 --name CelebAHQ_rdm  
```


## License
All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

## Acknowledgements
Codes and network architectures inspired from: <br>
- [DRIT++](https://github.com/HsinYingLee/MDMM)
- [StarGAN-v2](https://github.com/clovaai/stargan-v2)

