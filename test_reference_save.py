import torch
from options import TestOptions
from datasets import dataset_single
from model import SAVI2I
from saver import save_imgs
import os
import torchvision.utils as vutils

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  datasetA = dataset_single(opts, opts.index_s)
  datasetB = dataset_single(opts, opts.index_t)

  loader = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=1, shuffle=False)
  loader_attr = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=1, shuffle=True)


  # model
  print('\n--- load model ---')
  model = SAVI2I(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  # test
  print('\n--- testing ---')

  for idx1, img1  in enumerate(loader):
    print('{}/{}'.format(idx1, len(loader)))
    img1 = img1.cuda()

    for idx2, img2 in  enumerate(loader_attr):
      img2 = img2.cuda()
      if idx2>opts.num:
        break
      with torch.no_grad():
        imgs, names = model.test_interpolate_ref_save(img1, img2)
        imgs.append(img1.squeeze(0))
        imgs.append(img2.squeeze(0))
        names.append('input')
        names.append('reference')
        dir = os.path.join(result_dir, '{}'.format(idx1), '{}'.format(idx2))
        os.makedirs(dir, exist_ok=True)
        save_imgs(imgs, names, dir)
  return

if __name__ == '__main__':
  main()
