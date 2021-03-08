import os
import torchvision
import numpy as np
from PIL import Image

# tensor to PIL Image
def denormalize(x):
  out = (x + 1) / 2
  return out.clamp_(0, 1)

def tensor2img(img):
  img = denormalize(img)
  img = img.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
  return img.astype(np.uint8)


# save a set of images
def save_imgs(imgs, names, path):
  if not os.path.exists(path):
    os.mkdir(path)
  for img, name in zip(imgs, names):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(os.path.join(path, name + '.png'))

def save_concat_imgs(imgs, name, path):
  if not os.path.exists(path):
    os.mkdir(path)
  imgs = [tensor2img(i) for i in imgs]
  widths, heights,c = zip(*(i.shape for i in imgs))
  total_width = sum(widths)
  max_height = max(heights)
  new_im = Image.new('RGB', (total_width, max_height))
  x_offset = 0
  for im in imgs:
    im = Image.fromarray(im)
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  new_im.save(os.path.join(path, name + '.png'))

class Saver():
  def __init__(self, opts):
    self.model_dir = os.path.join(opts.result_dir, opts.name)
    self.image_dir = os.path.join(self.model_dir, 'images')
    self.img_save_freq = opts.img_save_freq
    self.model_save_freq = opts.model_save_freq

    # make directory
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
      os.makedirs(self.image_dir)


  # save result images
  def write_img(self, ep, model):
    if (ep + 1) % self.img_save_freq == 0:
      assembled_images = model.assemble_outputs()
      img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
      torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
    elif ep == -1:
      assembled_images = model.assemble_outputs()
      img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
      torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

  # save model
  def write_model(self, ep, total_it, model):
    if (ep + 1) % self.model_save_freq == 0:
      print('--- save the model @ ep %d ---' % (ep))
      model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
    elif ep == -1:
      model.save('%s/last.pth' % self.model_dir, ep, total_it)

