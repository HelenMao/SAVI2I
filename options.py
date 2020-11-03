import argparse

class BaseOptions(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--input_dim', type=int, default=3, help='# of input channels')
    self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
    self.parser.add_argument('--num_domains', type=int, default=2, help='number of visual domains')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--img_size', type=int, default=256, help='cropped image size for training')

    self.parser.add_argument('--input_nz', type=int, default=8, help='domain-specific attribute dimensions')
    self.parser.add_argument('--style_dim', type=int, default=64, help='output dimensions of fusing networks')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')

    self.parser.add_argument('--type', type=int, default=1,
                             help='set 1 for low-level style translation, set 0 for using shape-variation translation')






class TrainOptions(BaseOptions):
  def __init__(self):
    super(TrainOptions, self).__init__()

    self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')
    self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')

    # training related
    self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    self.parser.add_argument('--f_lr', type=float, default=1e-6, help='learning rate of fuse module')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=1200, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=600, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--lambda_rec', type=float, default=10.0)
    self.parser.add_argument('--lambda_r1', type=float, default=1.0, help='lambda of r1 regularization')
    self.parser.add_argument('--lambda_mmd', type=float, default=1.0, help='lambda of mmd loss')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions(BaseOptions):
  def __init__(self):
    super(TestOptions, self).__init__()
    self.parser.add_argument('--index_s', type=str, default='A', help='source domain index')
    self.parser.add_argument('--index_t', type=str, default='B', help='target domain index')
    self.parser.add_argument('--num', type=int, help='num of targets')
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='./outputs', help='path for saving result images and models')
    self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')


  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    # set irrelevant options
    self.opt.dis_scale = 3
    self.opt.dis_norm = 'None'
    self.opt.dis_spectral_norm = False
    return self.opt
