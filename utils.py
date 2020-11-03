import torch
import torch.nn as nn
from torch import autograd
import numpy as np

def compute_kernel(x, y):
  x_size = x.size(0)
  y_size = y.size(0)
  dim = x.size(1)
  x = x.unsqueeze(1) # (x_size, 1, dim)
  y = y.unsqueeze(0) # (1, y_size, dim)
  tiled_x = x.expand(x_size, y_size, dim)
  tiled_y = y.expand(x_size, y_size, dim)
  kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
  return torch.exp(-kernel_input) # (x_size, y_size)

def get_mmd_loss():
  def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd
  return compute_mmd



def calc_grad2(d_out, x_in):
  batch_size = x_in.size(0)
  grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
  grad_dout2 = grad_dout.pow(2)
  assert (grad_dout2.size() == x_in.size())
  reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
  return reg