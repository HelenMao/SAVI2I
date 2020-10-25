import os, sys
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import numpy as np
import torch

class dataset_multi(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot
    self.num_domains = opts.num_domains
    self.input_dim = opts.input_dim
    self.nz = opts.input_nz

    domains = [chr(i) for i in range(ord('A'),ord('Z')+1)]
    self.images = [None]*self.num_domains
    stats = ''
    for i in range(self.num_domains):
      img_dir = os.path.join(self.dataroot, opts.phase + domains[i])
      ilist = os.listdir(img_dir)
      self.images[i] = [os.path.join(img_dir, x) for x in ilist]
      stats += '{}: {}'.format(domains[i], len(self.images[i]))
    stats += ' images'
    self.dataset_size = max([len(self.images[i]) for i in range(self.num_domains)])

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.img_size))
    else:
      transforms.append(CenterCrop(opts.img_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)

    return

  def __getitem__(self, index):
    images = []
    labels = []
    label_mask = []
    label_id= []
    cls = torch.randperm(self.num_domains)
    #cls = random.randint(0,self.num_domains-1)
    for i in range(2):
      c_org = np.zeros((self.num_domains,))
      c_mask = np.ones((self.num_domains,)) * -1
      data = self.load_img(self.images[cls[i]][random.randint(0, len(self.images[cls[i]]) - 1)], self.input_dim)
      c_org[cls[i]] = 1
      c_mask[cls[i]] = 1
      images.append(data)
      c_org = torch.FloatTensor(c_org)
      labels.append(c_org)
      c_org_mask = torch.FloatTensor(c_mask)
      c_org_mask = c_org_mask.view(-1, 1).repeat(1,self.nz)
      c_org_mask = c_org_mask.view(self.num_domains*self.nz)
      label_mask.append(c_org_mask)
      label_id.append(cls[i])
    return images, labels, label_mask, label_id
  
  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size


class dataset_single(data.Dataset):
  def __init__(self, opts, domain):
    self.dataroot = opts.dataroot
    domains = [chr(i) for i in range(ord('A'),ord('Z')+1)]
    index = ord(domain) - ord('A')
    images = os.listdir(os.path.join(self.dataroot, opts.phase + domains[index]))
    self.img = [os.path.join(self.dataroot, opts.phase +  domains[index], x) for x in images]
    self.size = len(self.img)
    self.input_dim = opts.input_dim

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(CenterCrop(opts.img_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size
