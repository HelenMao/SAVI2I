import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision.models import vgg19
from collections import namedtuple
from copy import deepcopy
from functools import partial
import math
import torchvision.transforms as transforms


class E_content_shape(nn.Module):
  def __init__(self, img_size=256, input_dim=3, max_conv_dim=512):
    super(E_content_shape, self).__init__()
    enc_c = []
    dim_in = 2 ** 14 // img_size
    self.img_size = img_size
    enc_c += [nn.Conv2d(input_dim, dim_in, 3, 1, 1)]
    repeat_num = int(np.log2(img_size)) - 5
    for _ in range(repeat_num):
      dim_out = min(dim_in*2, max_conv_dim)
      enc_c += [ResBlk(dim_in, dim_out, normalize=True, downsample=True)]
      dim_in = dim_out

    for _ in range(2):
      enc_c += [ResBlk(dim_out, dim_out, normalize=True, downsample=False)]
    self.conv = nn.Sequential(*enc_c)

  def forward(self, x):
    return self.conv(x)

class E_content_style(nn.Module):
  def __init__(self, img_size=256, input_dim=3, max_conv_dim=256):
    super(E_content_style, self).__init__()
    enc_c = []
    dim_in = 2 ** 14 // img_size
    self.img_size = img_size
    enc_c += [nn.Conv2d(input_dim, dim_in, 3, 1, 1)]
    repeat_num = int(np.log2(img_size)) - 6
    for _ in range(repeat_num):
      dim_out = min(dim_in*2, max_conv_dim)
      enc_c += [ResBlk(dim_in, dim_out, normalize=True, downsample=True)]
      dim_in = dim_out

    for _ in range(2):
      enc_c += [ResBlk(dim_out, dim_out, normalize=True, downsample=False)]
    self.conv = nn.Sequential(*enc_c)

  def forward(self, x):
    return self.conv(x)

class E_attr(nn.Module):
  def __init__(self, img_size=256, input_dim=3, nz=8, n_domains=2, max_conv_dim=512):
    super(E_attr, self).__init__()
    dim_in = 2 ** 14 // img_size
    blocks = []
    blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

    repeat_num = int(np.log2(img_size)) - 2
    for _ in range(repeat_num):
      dim_out = min(dim_in * 2, max_conv_dim)
      blocks += [ResBlk(dim_in, dim_out, downsample=True)]
      dim_in = dim_out

    blocks += [nn.LeakyReLU(0.2)]
    blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
    blocks += [nn.LeakyReLU(0.2)]
    self.model = nn.Sequential(*blocks)
    self.fc = nn.Linear(dim_out, nz*n_domains)

  def forward(self, x):
    h = self.model(x).view(x.size(0), -1)
    output = self.fc(h)
    return output

class MappingNetwork(nn.Module):
  def __init__(self, nz=8, n_domains=2, n_style=64, hidden_dim=512, hidden_layer=1):
    super(MappingNetwork, self).__init__()
    layers = []
    layers += [nn.Linear(nz*n_domains, hidden_dim)]
    layers += [nn.ReLU()]
    for _ in range(hidden_layer):
      layers += [nn.Linear(hidden_dim, hidden_dim)]
      layers += [nn.ReLU()]

    layers+=[nn.Linear(hidden_dim, n_style)]

    self.model = nn.Sequential(*layers)

  def forward(self, latent_code):
    latent_code = self.model(latent_code)
    return latent_code

class Generator_shape(nn.Module):
  def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, output_dim=3):
    super(Generator_shape, self).__init__()
    dim_in = max_conv_dim
    repeat_num = int(np.log2(img_size)) - 5
    min_conv_dim = max_conv_dim // (2 ** repeat_num)
    self.decode = nn.ModuleList()
    self.to_rgb = nn.Sequential(
      nn.InstanceNorm2d(min_conv_dim, affine=True),
      nn.LeakyReLU(0.2),
      nn.Conv2d(min_conv_dim, output_dim, 1, 1, 0))

    for _ in range(2):
      dim_out = dim_in
      self.decode.append(AdainResBlk(dim_in, dim_out, style_dim))

    for _ in range(repeat_num):
      dim_in = dim_out
      dim_out = max(dim_in // 2, min_conv_dim)
      self.decode.append(AdainResBlk(dim_in, dim_out, style_dim, upsample=True))
    return

  def forward(self, x, s):
    for block in self.decode:
      x = block(x, s)
    return self.to_rgb(x)


class Generator_style(nn.Module):
  def __init__(self, img_size=256, style_dim=64, max_conv_dim=256, output_dim=3):
    super(Generator_style, self).__init__()
    dim_in = max_conv_dim
    repeat_num = int(np.log2(img_size)) - 6
    self.style_dim = style_dim
    min_conv_dim = max_conv_dim // (2 ** repeat_num)
    self.decode1 = nn.ModuleList()
    self.decode2 = nn.ModuleList()
    self.to_rgb = nn.Sequential(
        nn.InstanceNorm2d(min_conv_dim, affine=True),
        nn.LeakyReLU(0.2),
        nn.Conv2d(min_conv_dim, output_dim, 1, 1, 0))

    for _ in range(2):
        dim_out = dim_in
        self.decode1.append(ResBlk(dim_in + self.style_dim, dim_out, normalize=True))

    for _ in range(repeat_num):
        dim_in = dim_out
        dim_out = max(dim_in // 2, min_conv_dim)
        self.decode2.append(
            ReLUINSConvTranspose2d(dim_in + self.style_dim, dim_out, kernel_size=3, stride=2, padding=1,
                                   output_padding=1))

    self.to_rgb.apply(gaussian_weights_init)
    self.decode1.apply(he_init)
    return

  def forward(self, x, s):
    for block in self.decode1:
        s_img = s.view(s.size(0), s.size(1), 1, 1).expand(s.size(0), s.size(1), x.size(2), x.size(3))
        x_s = torch.cat([x, s_img], 1)
        x = block(x_s)
    for block in self.decode2:
        s_img = s.view(s.size(0), s.size(1), 1, 1).expand(s.size(0), s.size(1), x.size(2), x.size(3))
        x_s = torch.cat([x, s_img], 1)
        x = block(x_s)
    return self.to_rgb(x)

class Discriminator(nn.Module):
  def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
    super(Discriminator, self).__init__()
    dim_in = 2**14 // img_size
    self.num_domains = num_domains
    blocks = []
    blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]


    repeat_num = int(np.log2(img_size)) - 2

    for _ in range(repeat_num):
      dim_out = min(dim_in*2, max_conv_dim)
      blocks += [ResBlk(dim_in, dim_out, downsample=True)]
      dim_in = dim_out

    blocks += [nn.LeakyReLU(0.2)]
    blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
    blocks += [nn.LeakyReLU(0.2)]
    blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
    self.model = nn.Sequential(*blocks)


  def forward(self, x, domain_id):
    out = self.model(x)
    out = out.view(x.size(0),-1).view(x.size(0), self.num_domains, 1)
    out = gather_domain(out, domain_id).view(x.size(0),-1)
    return out

class Dis_content_shape(nn.Module):
  def __init__(self, c_dim=2, conv_dim=512):
    super(Dis_content_shape, self).__init__()
    model = []
    model += [ResBlk(conv_dim, conv_dim, downsample=True)]
    model += [ResBlk(conv_dim, conv_dim, downsample=True)]
    model += [ResBlk(conv_dim, conv_dim, downsample=True)]
    model += [nn.LeakyReLU(0.2)]
    model += [nn.Conv2d(conv_dim, conv_dim, 4, 1, 0)]
    model += [nn.LeakyReLU(0.2)]
    model += [nn.Conv2d(conv_dim, c_dim, 1, 1, 0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    out = out.view(out.size(0), out.size(1))
    return out

class Dis_content_style(nn.Module):
  def __init__(self, c_dim=2, conv_dim=256):
    super(Dis_content_style, self).__init__()
    model = []
    model += [ResBlk(conv_dim, conv_dim, downsample=True)]
    model += [ResBlk(conv_dim, conv_dim, downsample=True)]
    model += [ResBlk(conv_dim, conv_dim, downsample=True)]
    model += [ResBlk(conv_dim, conv_dim, downsample=True)]
    model += [nn.LeakyReLU(0.2)]
    model += [nn.Conv2d(conv_dim, conv_dim, 4, 1, 0)]
    model += [nn.LeakyReLU(0.2)]
    model += [nn.Conv2d(conv_dim, c_dim, 1, 1, 0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    out = out.view(out.size(0), out.size(1))
    return out



class VGG(object):
  def __init__(self, device):
    self.vgg = vgg19(pretrained=True).features
    self.vgg = self.vgg.to(device)
    self.device = device
    for param in self.vgg.parameters():
      param.requires_grad_(False)
    self.style_weights = {
    'conv1_1': 0.1,
    'conv2_1': 0.2,
    'conv3_1': 0.4,
    'conv4_1': 0.8,
    'conv5_1': 1.6,
    }

  def get_features(self, img, layers=None):
    """
    Use VGG19 to extract features from the intermediate layers.
    """
    if layers is None:
      layers = {
        '0': 'conv1_1',  # style layer
        '5': 'conv2_1',  # style layer
        '10': 'conv3_1',  # style layer
        '19': 'conv4_1',  # style layer
        '28': 'conv5_1',  # style layer

        '21': 'conv4_2'  # content layer
      }

    features = {}
    x = img
    for name, layer in self.vgg._modules.items():
      x = layer(x)
      if name in layers:
        features[layers[name]] = x

    return features

  def get_gram_matrix(self, img):
    """
       Compute the gram matrix by converting to 2D tensor and doing dot product
       img: (batch, channel, height, width)
       """
    b, c, h, w = img.size()
    img = img.view(b * c, h * w)
    gram = torch.mm(img, img.t())
    return gram

  def normalize(self, imgs):
    mean = torch.Tensor([0.485-1, 0.456-1, 0.406-1]).view(1, 3, 1, 1).to(self.device)
    #  if input in range [-1,1]
    std = torch.Tensor([0.229*2, 0.224*2, 0.225*2]).view(1, 3, 1, 1).to(self.device)
    #  if input in range [-1,1]
    norm_imgs = (imgs - mean) / std
    return norm_imgs

  def get_style_loss(self, source, target):
    source = self.normalize(source)
    target = self.normalize(target)
    source_features = self.get_features(source)
    target_feratures = self.get_features(target)
    style_loss = 0
    for layer in self.style_weights:
      source_feature = source_features[layer]
      target_feature = target_feratures[layer]
      source_gram = self.get_gram_matrix(source_feature)
      target_gram = self.get_gram_matrix(target_feature)
      layer_style_loss = self.style_weights[layer] * torch.mean((target_gram - source_gram) ** 2)
      b, c, h, w = target_feature.shape
      style_loss += layer_style_loss / (c * h * w)
    return style_loss

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def gather_domain(src, domain_index):  # only works at torch.gather(..., dim=1)
  domain_index = domain_index.view(domain_index.size(0), 1, 1)
  domain_index = domain_index.repeat(1, 1, src.size(-1)).long()
  output = torch.gather(src, 1, domain_index)
  return output

def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler

def he_init(module):
  if isinstance(module, nn.Conv2d):
    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    if module.bias is not None:
      nn.init.constant_(module.bias, 0)
  if isinstance(module, nn.Linear):
    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    if module.bias is not None:
      nn.init.constant_(module.bias, 0)

def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################
class ResBlk(nn.Module):
  def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
               normalize=False, downsample=False):
    super(ResBlk, self).__init__()
    self.actv = actv
    self.normalize = normalize
    self.downsample = downsample
    self.learned_sc = dim_in != dim_out
    self._build_weights(dim_in, dim_out)

  def _build_weights(self, dim_in, dim_out):
    self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
    self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
    if self.normalize:
      self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
      self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
    if self.learned_sc:
      self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

  def _shortcut(self, x):
    if self.learned_sc:
      x = self.conv1x1(x)
    if self.downsample:
      x = F.avg_pool2d(x, 2)
    return x

  def _residual(self, x):
    if self.normalize:
      x = self.norm1(x)
    x = self.actv(x)
    x = self.conv1(x)
    if self.downsample:
      x = F.avg_pool2d(x, 2)
    if self.normalize:
      x = self.norm2(x)
    x = self.actv(x)
    x = self.conv2(x)
    return x

  def forward(self, x):
    x = self._shortcut(x) + self._residual(x)
    return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
  def __init__(self, style_dim, num_features):
    super(AdaIN, self).__init__()
    self.norm = nn.InstanceNorm2d(num_features, affine=False)
    self.fc = nn.Linear(style_dim, num_features * 2)

  def forward(self, x, s):
    h = self.fc(s)
    h = h.view(h.size(0), h.size(1), 1, 1)
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
  def __init__(self, dim_in, dim_out, style_dim=64,
               actv=nn.LeakyReLU(0.2), upsample=False):
    super(AdainResBlk, self).__init__()
    self.actv = actv
    self.upsample = upsample
    self.learned_sc = dim_in != dim_out
    self._build_weights(dim_in, dim_out, style_dim)

  def _build_weights(self, dim_in, dim_out, style_dim=64):
    self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
    self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
    self.norm1 = AdaIN(style_dim, dim_in)
    self.norm2 = AdaIN(style_dim, dim_out)
    if self.learned_sc:
      self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

  def _shortcut(self, x):
    if self.upsample:
      x = F.interpolate(x, scale_factor=2, mode='nearest')
    if self.learned_sc:
      x = self.conv1x1(x)
    return x

  def _residual(self, x, s):
    x = self.norm1(x, s)
    x = self.actv(x)
    if self.upsample:
      x = F.interpolate(x, scale_factor=2, mode='nearest')
    x = self.conv1(x)
    x = self.norm2(x, s)
    x = self.actv(x)
    x = self.conv2(x)
    return x

  def forward(self, x, s):
    out = self._residual(x, s)
    out = (out + self._shortcut(x)) / math.sqrt(2)
    return out


class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                   output_padding=output_padding, bias=True)]
    model += [LayerNorm(n_out)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    return

  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape),
                            self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

