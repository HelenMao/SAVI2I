import torch
from options import TestOptions
from datasets import dataset_single
from model import SAVI2I
from saver import save_imgs, save_concat_imgs
import os

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()


  # data loader
  print('\n--- load dataset ---')
  dataset = dataset_single(opts, opts.index_s)
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

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

  for idx, img in enumerate(loader):
    #break
    img = img.cuda()
    for idx2 in range(opts.num):
      with torch.no_grad():
        index = ord(opts.index_t) - ord('A')
        imgs, names = model.test_interpolate_latent_save_rdm(img, index)
        dir = os.path.join(result_dir, '{}'.format(idx), '{}'.format(idx2))
        os.makedirs(dir, exist_ok=True)
        save_imgs(imgs, names, dir)
  return

if __name__ == '__main__':
  main()
