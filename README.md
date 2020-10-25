

# SAVI2I: Continuous and Diverse Image-to-Image Translation via Signed Attribute Vectors

<img src='imgs/teasear.png' width="900px">

Pytorch implementation for SAVI2I. We propose a simple yet effective signed attribute vectors (SAV) that facilitates **continuous** translation on **diverse** mapping paths across **multiple** domains. 

## Qualitative Results
### Reference-guided
- Summer2Winter

- Photo2Artwork

- Male2Female

- Female2Male

- AFHQ


## Usage

### Prerequisites
- Python 3.5 or Python 3.6
- Pytorch 0.4.0+


### Install
- Clone this repo:
```
git clone https://github.com/HelenMao/SAVI2I.git
```
## Training Datasets
Download datasets for each task into the dataset folder
```
mkdir datasets
```
- Yosemite  (summer <-> winter) 
- Photo2Artworks
- CelebA-HQ
We split CelebA-HQ into male and female domains according to the annotated label and fine-tune the images manaully. 
- AFHQ 

## Training
- Yosemite
```
python train.py --dataroot ./datasets/Yosemite/ --phase train --type 1 --name yosemite --n_ep 1000 --n_ep_decay 500 --lambda_r1 10 --lambda_mmd 1 --num_domains 2
```
- Photo2Artworks
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


Download and save them into 
```
./models/

```

### Testing Examples
**Reference-guided
```
python test_reference_save.py --dataroot ./datasets/CelebAHQ --resume ./models/CelebAHQ/00029.pth
```
**Pix2Pix-Mode-Seeking** <br>
```
python test.py --dataroot ./datasets/facades --checkpoints_dir ./models/Pix2Pix-Mode-Seeking/facades --epoch 400
```
```
python test.py --dataroot ./datasets/maps --checkpoints_dir ./models/Pix2Pix-Mode-Seeking/maps --epoch 400
```
**DRIT-Mode-Seeking** <br>
```
python test.py --dataroot ./datasets/yosemite --resume ./models/DRIT-Mode-Seeking/yosemite/01200.pth --concat 1
```
```
python test.py --dataroot ./datasets/cat2dog --resume ./models/DRIT-Mode-Seeking/cat2dog/01999.pth --concat 0
```
**StackGAN++-Mode-Seeking** <br>
```
python main.py --cfg cfg/eval_birds.yml 
```
