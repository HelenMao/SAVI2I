[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)


# SAVI2I: Continuous and Diverse Image-to-Image Translation via Signed Attribute Vectors
### [[Paper](https://arxiv.org/abs/2011.01215)] [[Project Website]](https://helenmao.github.io/SAVI2I/)
<img src='imgs/teaser.png' width="1200px">

Pytorch implementation for SAVI2I. We propose a simple yet effective signed attribute vector (SAV) that facilitates **continuous** translation on **diverse** mapping paths across **multiple** domains. 
<br>
More video results please see [Our Webpage](https://helenmao.github.io/SAVI2I/)
<br>
Contact: Qi Mao (qimao@pku.edu.cn)

## Paper
Continuous and Diverse Image-to-Image Translation via Signed Attribute Vectors<br>
[Qi Mao](https://sites.google.com/view/qi-mao/), [Hsin-Ying Lee](https://research.snap.com/team/hsin-ying-lee/), [Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), [Siwei Ma](https://scholar.google.com/citations?user=y3YqlaUAAAAJ&hl=zh-CN), and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)<br>
In [arXiv 2020](https://arxiv.org/abs/2011.01215)

## Citation
If you find this work useful for your research, please cite our paper:

```
    @article{mao2020continuous,
      author       = "Mao, Qi and Lee, Hsin-Ying and Tseng, Hung-Yu and Huang, Jia-Bin and Ma, Siwei and Yang, Ming-Hsuan",
      title        = "Continuous and Diverse Image-to-Image Translation via Signed Attribute Vectors",
      journal    = "arXiv preprint 2011.01215",
      year         = "2020"
    }
```

## Quick Start
### Prerequisites
- Linux or Windows
- Python 3+
- Suggest to use two P100 16GB GPUs or One V100 32GB GPU.


### Install
- Clone this repo:
```bash
git clone https://github.com/HelenMao/SAVI2I.git
cd SAVI2I
```
- This code requires Pytorch 0.4.0+ and Python 3+. Please install dependencies by
```bash
conda create -n SAVI2I python=3.6
source activate SAVI2I
pip install -r requirements.txt 
```

### Training Datasets
Download datasets for each task into the dataset folder
```
./datasets
```
- Style translation: Yosemite  (summer <-> winter) and Photo2Artwork (Photo, Monet, Van Gogh and Ukiyo-e) <br>
>* You can follow the instructions of [CycleGAN datasets](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md) to download Yosemite and Photo2artwork datasets.

- Shape-variation translation: CelebA-HQ (Male  <-> Female)  and AFHQ (Cat, Dog and WildLife) <br>
>* We split CelebA-HQ into male and female domains according to the annotated label and fine-tune the images manaully. 
>* You can follow the instructions of [StarGAN-v2 datasets](https://github.com/clovaai/stargan-v2) to download CelebA-HQ and AFHQ datasets.

## Training

### Notes
> For low-level style translation tasks, you suggest to set ```--type=1``` to use corresponding network architectures. <br>
> For shape-variation translation tasks, you suggest to set ```--type=0``` to use corresponding network architectures.


- Yosemite
```bash
python train.py --dataroot ./datasets/Yosemite/ --phase train --type 1 --name Yosemite --n_ep 700 --n_ep_decay 500 --lambda_r1 10 --lambda_mmd 1 --num_domains 2
```
- Photo2artwork
```bash
python train.py --dataroot ./datasets/Photo2artwork/ --phase train --type 1 --name Photo2artwork --n_ep 100 --n_ep_decay 0 --lambda_r1 10 --lambda_mmd 1 --num_domains 4
```
- CelebAHQ
```bash
python train.py --dataroot ./datasets/CelebAHQ/ --phase train --type 0 --name CelebAHQ --n_ep 30 --n_ep_decay 0 --lambda_r1 1 --lambda_mmd 1 --num_domains 2
```
- AFHQ
```bash
python train.py --dataroot ./datasets/AFHQ/ --phase train --type 0 --name AFHQ --n_ep 100 --n_ep_decay 0 --lambda_r1 1 --lambda_mmd 10 --num_domains 3
```


## Pre-trained Models
- [Yosemite](https://drive.google.com/file/d/1relOFLfOW0ACpr_u6DXgll7Qf6sLSstx/) 
- [Photo2artwork](https://drive.google.com/file/d/1B1G_Ml-a0phvBG_ePlbwBOpT4hOk9X9h/)
- [CelebAHQ](https://drive.google.com/file/d/1x0sRX-QTQ3z5Eep-ROmX9wcAMHWIBT6j/)
- [AFHQ](https://drive.google.com/file/d/19hsK63GJyT_qaqAwaE8mM_zBWncYtBzB/)

Download and save them into 
```
./models
```
or download the pre-trained models with the following script.
```bash
bash ./download_models.sh
```

## Testing 
**Reference-guided**
```bash
python test_reference_save.py --dataroot ./datasets/CelebAHQ --resume ./models/CelebAHQ/00029.pth --phase test --type 0 --num_domains 2 --index_s A --index_t B --num 5 --name CelebAHQ_ref  
```
**Latent-guided** 
```bash
python test_latent_rdm_save.py --dataroot ./datasets/CelebAHQ --resume ./models/CelebAHQ/00029.pth --phase test --type 0 --num_domains 2 --index_s A --index_t B --num 5 --name CelebAHQ_rdm  
```


## License
All rights reserved. <br>
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**). <br>
The codes are only for academical research use. For commercial use, please contact qimao@pku.edu.cn.

## Acknowledgements
Codes and network architectures inspired from: <br>
- [DRIT++](https://github.com/HsinYingLee/MDMM)
- [StarGAN-v2](https://github.com/clovaai/stargan-v2)

