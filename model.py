import torch
import torch.nn as nn
import numpy as np
import utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import torchvision.utils as vutils
import networks

class SAVI2I(nn.Module):
  def __init__(self, opts):
    super(SAVI2I, self).__init__()
    self.opts = opts
    if opts.gpu >= 0:
        self.device = torch.device('cuda:%d' % opts.gpu)
    else:
        self.device = torch.device('cpu')
        torch.cuda.set_device(opts.gpu)
        cudnn.benchmark = True

    self.phase = opts.phase
    self.type = opts.type
    self.nz = opts.input_nz
    self.style_dim = opts.style_dim
    self.num_domains = opts.num_domains


    self.enc_a = nn.DataParallel(
        networks.E_attr(img_size=opts.img_size, input_dim=opts.input_dim, nz=self.nz, n_domains=self.num_domains))
    self.f = nn.DataParallel(
        networks.MappingNetwork(nz=self.nz, n_domains=self.num_domains, n_style=self.style_dim, hidden_dim=512,
                                hidden_layer=1))
    self.vgg = networks.VGG(self.device)
    if self.type==1:
      self.enc_c = nn.DataParallel(networks.E_content_style(img_size=opts.img_size, input_dim=opts.input_dim))
      self.gen = nn.DataParallel(networks.Generator_style(img_size=opts.img_size, style_dim=self.style_dim))
    elif self.type==0:
      self.enc_c = nn.DataParallel(networks.E_content_shape(img_size=opts.img_size, input_dim=opts.input_dim))
      self.gen = nn.DataParallel(networks.Generator_shape(img_size=opts.img_size, style_dim=self.style_dim))

    if self.phase == 'train':
      self.lr = opts.lr
      self.f_lr = opts.f_lr
      self.lr_dcontent = self.lr/2.5
      self.dis = nn.DataParallel(networks.Discriminator(img_size=opts.img_size, num_domains=self.num_domains))
      if self.type==1:
        self.disContent = nn.DataParallel(networks.Dis_content_style(c_dim=self.num_domains))
      elif self.type==0:
        self.disContent = nn.DataParallel(networks.Dis_content_shape(c_dim=self.num_domains))
      self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.lr, betas=(0, 0.99), weight_decay=0.0001)
      self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=self.lr, betas=(0, 0.99), weight_decay=0.0001)
      self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=self.lr, betas=(0, 0.99), weight_decay=0.0001)
      self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0, 0.99), weight_decay=0.0001)
      self.f_opt = torch.optim.Adam(self.f.parameters(), lr=self.f_lr, betas=(0, 0.99), weight_decay=0.0001)
      self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=self.lr_dcontent, betas=(0, 0.99), weight_decay=0.0001)

      self.criterion_GAN = nn.BCEWithLogitsLoss()
      self.criterion_mmd = utils.get_mmd_loss()

  def initialize(self):
    self.dis.apply(networks.he_init)
    if self.type == 0:
      self.gen.apply(networks.he_init)
    self.enc_c.apply(networks.he_init)
    self.enc_a.apply(networks.he_init)
    self.f.apply(networks.he_init)
    self.disContent.apply(networks.he_init)

  def set_scheduler(self, opts, last_ep=0):
    self.dis_sch = networks.get_scheduler(self.dis_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)
    self.f_sch = networks.get_scheduler(self.f_opt, opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)

  def update_lr(self):
    self.dis_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()
    self.f_sch.step()
    self.disContent_sch.step()

  def setgpu(self, gpu):
    self.gpu = gpu
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)
    self.f.cuda(self.gpu)
    if self.phase=='train':
      self.disContent.cuda(self.gpu)
      self.dis.cuda(self.gpu)

  def get_z_random(self, batchSize, nz):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z


  def forward(self):
    # input images
    half_size = self.input.size(0)//2
    self.real_A = self.input[0:half_size]
    self.real_B = self.input[half_size:]
    c_org_A = self.c_org[0:half_size]
    c_org_B = self.c_org[half_size:]

    c_org_mask_A = self.c_org_mask[0:half_size]
    c_org_mask_B = self.c_org_mask[half_size:]

    # get encoded z_c
    self.real_img = torch.cat((self.real_A, self.real_B),0)
    self.z_content = self.enc_c.forward(self.real_img)
    self.z_content_a, self.z_content_b = torch.split(self.z_content, half_size, dim=0)

    # get encoded z_a
    self.z_attr = self.enc_a.forward(self.real_img)
    self.z_attr_a, self.z_attr_b = torch.split(self.z_attr, half_size, dim=0)

    # get random z_a
    self.z_random = self.get_z_random(self.input.size(0), self.nz*self.num_domains)
    self.z_random = torch.abs(self.z_random) * self.c_org_mask
    self.z_random_a, self.z_random_b = torch.split(self.z_random, half_size, dim=0)

    self.z_random2 = self.get_z_random(self.input.size(0), self.nz*self.num_domains)
    self.z_random2 = torch.abs(self.z_random2) * self.c_org_mask
    self.z_random_a2, self.z_random_b2 = torch.split(self.z_random2, half_size, dim=0)


  # reverse codes
    rvs_index = torch.LongTensor(range(-half_size, half_size)).cuda()
    self.input_rvs = self.input[rvs_index]
    self.c_rvs = self.c_org[rvs_index]
    self.c_rvs_mask = self.c_org_mask[rvs_index]
    self.c_rvs_id = self.c_org_id[rvs_index]

    self.z_random_rvs = self.z_random* self.c_org_mask * self.c_rvs_mask
    self.z_random_rvs_a, self.z_random_rvs_b = torch.split(self.z_random_rvs, half_size, dim=0)

    # interpolation between source and reverse codes
    alpha = torch.rand(self.z_attr_a.size(0), 1).cuda(self.gpu)
    self.alpha = torch.cat((alpha, alpha), 0)
    self.z_interp_a2rvs = self.z_random_a + alpha * (self.z_random_rvs_a - self.z_random_a)
    self.z_interp_b2rvs = self.z_random_b + alpha * (self.z_random_rvs_b - self.z_random_b)

    # first cross translation
    input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b, self.z_content_b, self.z_content_b, self.z_content_b), 0)
    input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a, self.z_content_a, self.z_content_a, self.z_content_a), 0)
    input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random_a, self.z_random_rvs_a, self.z_interp_a2rvs, self.z_random_a2), 0)
    input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random_b, self.z_random_rvs_b, self.z_interp_b2rvs, self.z_random_b2), 0)

    input_attr_forA = self.f.forward(input_attr_forA)
    input_attr_forB = self.f.forward(input_attr_forB)

    output_fakeA = self.gen.forward(input_content_forA, input_attr_forA)
    output_fakeB = self.gen.forward(input_content_forB, input_attr_forB)
    self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random, self.fake_A_random_rvs, self.fake_A_interp, self.fake_A_random2 = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
    self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random, self.fake_B_random_rvs, self.fake_B_interp, self.fake_B_random2 = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # get reconstructed encoded z_c
    self.fake_encoded_img = torch.cat((self.fake_A_encoded, self.fake_B_encoded), 0)
    self.fake_random_img = torch.cat((self.fake_A_random, self.fake_B_random), 0)
    self.z_content_recon = self.enc_c.forward(torch.cat((self.fake_encoded_img, self.fake_random_img),0))
    self.z_content_recon_b, self.z_content_recon_a, self.z_content_recon_b2, self.z_content_recon_a2 = torch.split(self.z_content_recon, half_size, dim=0)


    # cycle consistency translation
    self.fake_A_recon = self.gen.forward(torch.cat((self.z_content_recon_a, self.z_content_recon_a2),0), self.f.forward(torch.cat((self.z_attr_a,self.z_attr_a),0)))
    self.fake_B_recon = self.gen.forward(torch.cat((self.z_content_recon_b, self.z_content_recon_b2),0), self.f.forward(torch.cat((self.z_attr_b,self.z_attr_b),0)))
    self.fake_A_recon1, self.fake_A_recon2 = torch.split(self.fake_A_recon, half_size, dim=0)
    self.fake_B_recon1, self.fake_B_recon2 = torch.split(self.fake_B_recon, half_size, dim=0)


    # for latent regression
    self.fake_random_img2 = torch.cat((self.fake_A_random2, self.fake_B_random2), 0)
    self.fake_random_rvs_img = torch.cat((self.fake_A_random_rvs, self.fake_B_random_rvs), 0)
    self.z_content_random_rvs_recon = self.enc_c.forward(self.fake_random_rvs_img)
    self.z_attr_random = self.enc_a.forward(self.fake_random_img)
    self.z_attr_random_a, self.z_attr_random_b = torch.split(self.z_attr_random, half_size, 0)
    self.fake_interp_img = torch.cat((self.fake_A_interp, self.fake_B_interp))



  def update_D_content(self, image, c_org):
    self.input = image
    self.z_content = self.enc_c.forward(self.input)
    self.disContent_opt.zero_grad()
    pred_cls = self.disContent.forward(self.z_content.detach())
    loss_D_content = self.criterion_GAN(pred_cls, c_org)
    loss_D_content.backward()
    self.disContent_loss = loss_D_content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image, c_org, c_org_mask, c_org_id):
    self.input = image
    self.c_org = c_org
    self.c_org_mask = c_org_mask
    self.c_org_id = c_org_id
    self.forward()

    self.dis_opt.zero_grad()
    self.D1_gan_loss, self.D1_reg_loss = self.backward_D(self.dis, self.input, self.fake_encoded_img, self.c_org_id)


    self.D2_gan_loss, self.D2_reg_loss = self.backward_D(self.dis, self.input, self.fake_random_img, self.c_org_id)
    loss_D2_rvs_loss, D2_reg_rvs_loss = self.backward_D(self.dis, self.input_rvs, self.fake_random_rvs_img, self.c_rvs_id)
    loss_D2_gan_loss2, loss_D2_reg_loss2 = self.backward_D(self.dis, self.input, self.fake_random_img2, self.c_org_id)
    if self.alpha[0]<0.5:
      loss_D2_interp_loss, D2_reg_interp_loss = self.backward_D(self.dis, self.input, self.fake_interp_img, self.c_org_id)
    elif self.alpha[0] == 0.5:
      loss_D2_interp_loss1, D2_reg_interp_loss1 = self.backward_D_mix(self.dis, self.input, self.fake_interp_img, self.c_org_id)
      loss_D2_interp_loss2, D2_reg_interp_loss2 = self.backward_D_mix(self.dis, self.input_rvs, self.fake_interp_img, self.c_rvs_id)
      loss_D2_interp_loss = loss_D2_interp_loss1 + loss_D2_interp_loss2
      D2_reg_interp_loss = D2_reg_interp_loss1 + D2_reg_interp_loss2
    else:
      loss_D2_interp_loss, D2_reg_interp_loss = self.backward_D(self.dis, self.input_rvs, self.fake_interp_img, self.c_rvs_id)

    self.D2_gan_loss += loss_D2_gan_loss2 + loss_D2_rvs_loss + loss_D2_interp_loss
    self.D2_reg_loss += loss_D2_reg_loss2 + D2_reg_rvs_loss + D2_reg_interp_loss
    self.dis_opt.step()

  def backward_D(self, netD, real, fake, c_label):
    real.requires_grad_()
    pred_fake  = netD.forward(fake.detach(), c_label)
    pred_real  = netD.forward(real, c_label)
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = out_a
      out_real = out_b
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = self.criterion_GAN(out_fake, all0)
      ad_true_loss = self.criterion_GAN(out_real, all1)
      loss_D += ad_true_loss + ad_fake_loss
    loss_D.backward(retain_graph=True)
    l_reg = utils.calc_grad2(pred_real, real) *self.opts.lambda_r1
    l_reg.backward()
    return loss_D.item(), l_reg.item()

  def backward_D_mix(self, netD, real, fake, c_label):
    real.requires_grad_()
    pred_fake = netD.forward(fake.detach(), c_label)
    pred_real = netD.forward(real, c_label)
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = out_a
      out_real = out_b
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = self.criterion_GAN(out_fake, all0)
      ad_true_loss = self.criterion_GAN(out_real, all1)
      loss_D += (ad_true_loss + ad_fake_loss) * 0.5
    loss_D.backward(retain_graph=True)
    l_reg = utils.calc_grad2(pred_real, real) * 0.5 * self.opts.lambda_r1
    l_reg.backward(retain_graph=True)
    return loss_D.item(), l_reg.item()

  def update_EFG(self):
    # update G, Ec, Ea, F
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.f_opt.zero_grad()
    self.backward_EFG()
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()
    self.f_opt.step()



  def backward_EFG(self):
    #Domain Adversarial loss
    pred_fake = self.dis.forward(self.fake_encoded_img, self.c_org_id)
    all_ones = torch.ones_like(pred_fake).cuda(self.gpu)
    loss_G_GAN = self.criterion_GAN(pred_fake, all_ones)

    pred_fake = self.dis.forward(self.fake_random_img, self.c_org_id)
    loss_G_GAN2 = self.criterion_GAN(pred_fake, all_ones)

    pred_fake_rvs = self.dis.forward(self.fake_random_rvs_img, self.c_rvs_id)
    loss_G_GAN2 += self.criterion_GAN(pred_fake_rvs, all_ones)

    pred_fake2 = self.dis.forward(self.fake_random_img2, self.c_org_id)
    loss_G_GAN2 += self.criterion_GAN(pred_fake2, all_ones)
    
    #Domain loss for interpolation results
    if self.alpha[0] < 0.5:
      pred_fake = self.dis.forward(self.fake_interp_img, self.c_org_id)
      loss_G_GAN2 += self.criterion_GAN(pred_fake, all_ones)
    elif self.alpha[0] == 0.5:
      pred_fake1 = self.dis.forward(self.fake_interp_img, self.c_org_id)
      loss_G_GAN2 += self.criterion_GAN(pred_fake1, all_ones) * 0.5
      pred_fake2 = self.dis.forward(self.fake_interp_img, self.c_rvs_id)
      loss_G_GAN2 += self.criterion_GAN(pred_fake2, all_ones) * 0.5
    else:
      pred_fake = self.dis.forward(self.fake_interp_img, self.c_rvs_id)
      loss_G_GAN2 += self.criterion_GAN(pred_fake, all_ones)

    # content Ladv for generator
    loss_G_GAN_content = self.backward_G_GAN_content(self.z_content)

    # self and cross-cycle recon
    loss_G_L1_self = torch.mean(torch.abs(self.input - torch.cat((self.fake_AA_encoded, self.fake_BB_encoded), 0))) * self.opts.lambda_rec
    loss_G_L1_cc = torch.mean(torch.abs(torch.cat((self.input,self.input),0) - torch.cat((self.fake_A_recon, self.fake_B_recon), 0))) * self.opts.lambda_rec

    #Style loss
    loss_style = self.vgg.get_style_loss(self.fake_encoded_img, self.input)

    # mmd loss
    loss_mmd_za = (self.criterion_mmd(self.z_attr_a, self.z_random_a) + self.criterion_mmd(self.z_attr_b, self.z_random_b))* self.opts.lambda_mmd

    # latent regression loss
    loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random_a)) * self.opts.lambda_rec
    loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random_b)) * self.opts.lambda_rec

    # mode seeking loss for A-->B and B-->A
    lz_AB = torch.mean(torch.abs(self.fake_B_random2 - self.fake_B_random)) / torch.mean(
      torch.abs(self.z_random_b2 - self.z_random_b))
    lz_BA = torch.mean(torch.abs(self.fake_A_random2 - self.fake_A_random)) / torch.mean(
      torch.abs(self.z_random_a2 - self.z_random_a))
    eps = 1 * 1e-5
    loss_lz_AB = 1 / (lz_AB + eps)
    loss_lz_BA = 1 / (lz_BA + eps)

   
    loss_G = loss_G_GAN+ loss_G_GAN2 + loss_G_L1_self + loss_G_L1_cc + loss_mmd_za + loss_style + loss_G_GAN_content +\
             loss_z_L1_a + loss_z_L1_b + loss_lz_AB + loss_lz_BA 
    loss_G.backward()

    self.gan_loss = loss_G_GAN.item() + loss_G_GAN2.item()
    self.gan_loss_content = loss_G_GAN_content.item()
    self.mmd_loss_za = loss_mmd_za.item()
    self.l1_self_rec_loss = loss_G_L1_self.item()
    self.l1_cc_rec_loss = loss_G_L1_cc.item()
    self.style_loss = loss_style.item()
    self.l1_recon_z_loss = loss_z_L1_a.item() + loss_z_L1_b.item()
    self.lz_ms = loss_lz_AB.item() + loss_lz_BA.item()
    self.G_loss = loss_G.item()

  def backward_G_GAN_content(self, data):
    pred_cls = self.disContent.forward(data)
    loss_G_content = self.criterion_GAN(pred_cls, 1-self.c_org)
    return loss_G_content

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A).detach()
    images_b = self.normalize_image(self.real_B).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    images_a2 = self.normalize_image(self.fake_A_random).detach()
    images_a3 = self.normalize_image(self.fake_A_recon1).detach()
    images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
    images_a5 = self.normalize_image(self.fake_A_random_rvs).detach()
    images_a6 = self.normalize_image(self.fake_A_interp).detach()
    images_a7 = self.normalize_image(self.fake_A_recon2).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    images_b2 = self.normalize_image(self.fake_B_random).detach()
    images_b3 = self.normalize_image(self.fake_B_recon1).detach()
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
    images_b5 = self.normalize_image(self.fake_B_random_rvs).detach()
    images_b6 = self.normalize_image(self.fake_B_interp).detach()
    images_b7 = self.normalize_image(self.fake_B_recon2).detach()
    row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_b6[0:1, ::], images_b5[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::], images_a7[0:1, ::]), 3)
    row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_a6[0:1, ::], images_a5[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::], images_b7[0:1, ::]), 3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]

  def save(self, filename, ep, total_it):
    state = {
      'dis': self.dis.module.state_dict(),
      'disContent': self.disContent.module.state_dict(),
      'enc_c': self.enc_c.module.state_dict(),
      'enc_a': self.enc_a.module.state_dict(),
      'gen': self.gen.module.state_dict(),
      'f': self.f.module.state_dict(),
      'dis_opt': self.dis_opt.state_dict(),
      'disContent_opt': self.disContent_opt.state_dict(),
      'enc_c_opt': self.enc_c_opt.state_dict(),
      'enc_a_opt': self.enc_a_opt.state_dict(),
      'gen_opt': self.gen_opt.state_dict(),
      'f_opt': self.f_opt.state_dict(),
      'ep': ep,
      'total_it': total_it
    }

    torch.save(state, filename)
    return

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    if train:
      self.dis.module.load_state_dict(checkpoint['dis'])
      self.disContent.module.load_state_dict(checkpoint['disContent'])

    self.enc_c.module.load_state_dict(checkpoint['enc_c'])
    self.enc_a.module.load_state_dict(checkpoint['enc_a'])
    self.gen.module.load_state_dict(checkpoint['gen'])
    self.f.module.load_state_dict(checkpoint['f'])
    # optimizer
    if train:
      self.dis_opt.load_state_dict(checkpoint['dis_opt'])
      self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
      self.f_opt.load_state_dict(checkpoint['f_opt'])
    return checkpoint['ep'], checkpoint['total_it']


  def test_interpolate_latent_save_rdm(self, img, index):
    z_content = self.enc_c.forward(img)
    latent_code1 = self.enc_a.forward(img)

    c_mask = np.ones((self.num_domains,)) * -1
    c_mask[index] = 1
    c_trg_mask = torch.FloatTensor(c_mask).cuda()
    c_trg_mask = c_trg_mask.view(-1, 1).repeat(1, self.nz)
    c_trg_mask = c_trg_mask.view(self.num_domains * self.nz)

    latent_code2 = self.get_z_random(img.size(0), self.nz * self.num_domains)
    latent_code2 = torch.abs(latent_code2) * c_trg_mask

    alpha_list = np.linspace(0, 1, 20)
    alpha = torch.FloatTensor([alpha_list]).cuda().view(-1,1)
    latent_code1 = latent_code1.repeat(alpha.size(0),1)
    latent_code2 = latent_code2.repeat(alpha.size(0),1)
    latent_code_mix = latent_code1 + alpha * (latent_code2 - latent_code1)
    z_content = z_content.repeat(alpha.size(0),1,1,1)
    output = self.gen.forward(z_content, self.f.forward(latent_code_mix))
    output_list = [output[i] for i in range(output.size(0))]
    names = ['output_{}'.format(i) for i in range(output.size(0))]
    return output_list, names


  def test_interpolate_ref_save(self, img1, img2):
    z_content = self.enc_c.forward(img1)
    latent_code1 = self.enc_a.forward(img1)
    latent_code2 = self.enc_a.forward(img2)
    alpha_list = np.linspace(0, 1, 21)
    alpha = torch.FloatTensor([alpha_list]).cuda().view(-1, 1)
    latent_code1 = latent_code1.repeat(alpha.size(0), 1)
    latent_code2 = latent_code2.repeat(alpha.size(0), 1)
    latent_code_mix = latent_code1 + alpha * (latent_code2 - latent_code1)
    z_content = z_content.repeat(alpha.size(0), 1, 1, 1)
    output = self.gen.forward(z_content, self.f.forward(latent_code_mix))
    output_list = [output[i] for i in range(output.size(0))]
    names = ['output_{}'.format(i) for i in range(output.size(0))]
    return output_list, names
