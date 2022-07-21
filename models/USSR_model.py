import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
import cv2
import numpy as np
from utils import util

from torch.nn.parallel import DataParallel



logger = logging.getLogger('base')


class USSR_Model(BaseModel):
    def __init__(self, opt):
        super(USSR_Model, self).__init__(opt)

        self.hmap = {
            'eta': None,
            'cost_list': None,
            'hmap_list': [],
            'hmap_x_list': [],      # patch_imscores
            'hmap_y_list': [],      # patch_psnrs
            'hmap_y_mean_list': [],      # patch_psnrs
            'bins': None,

        }


        if opt['test_mode'] == 'image':
            self.patch_size = int(opt["patch_size"])
            self.step = int(opt["step"])
        self.scale = int(opt["scale"])
        self.name = opt['name']
        self.which_model = opt['network_G']['which_model_G']

        train_opt = opt['train']

        width_list = np.array(opt['network_G']['width_list'])
        nf = np.array(opt['network_G']['nf'])
        self.mult_list = width_list / nf


        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

        if opt['dist'] == 'dp':
            self.netG = DataParallel(self.netG)

        # print network
        self.print_network()
        # self.load()

        self.pf = opt['logger']['print_freq']

        self.batch_size = int(opt['datasets']['train']['batch_size'])
        # self.batch_size = int(opt['datasets']['train_1']['batch_size'])

        self.netG.train()

        # loss
        loss_type = train_opt['pixel_criterion']
        if loss_type == 'l1':
            self.cri_pix = nn.L1Loss().to(self.device)
        elif loss_type == 'l2':
            self.cri_pix = nn.MSELoss().to(self.device)
        elif loss_type == 'cb':
            self.cri_pix = CharbonnierLoss().to(self.device)
        else:
            raise NotImplementedError('Loss _type [{:s}] is not recognized.'.format(loss_type))

        # optimizers
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_G)

        # schedulers
        if train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'], restarts=train_opt['restarts'], weights=train_opt['restart_weights'])
                )
        else:
            raise NotImplementedError

        self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)
        self.LQ_path = data['LQ_path'][0]

        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
            self.GT_path = data['GT_path'][0]

    def test_patch(self):
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)


    def test_image(self):
        self.netG.eval()
        self.var_L = cv2.imread(self.LQ_path, cv2.IMREAD_UNCHANGED)
        self.real_H = cv2.imread(self.GT_path, cv2.IMREAD_UNCHANGED)

        lr_list, num_h, num_w, h, w = self.crop_cpu(self.var_L, self.patch_size, self.step)
        gt_list = self.crop_cpu(self.real_H, self.patch_size*4, self.step*4)[0]
        sr_list = []
        index = 0

        psnr_type = [0] * (len(self.opt['network_G']['width_list']))

        for LR_img, GT_img in zip(lr_list, gt_list):
            img = LR_img.astype(np.float32) / 255.
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            # some images have 4 channels
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img = img[:, :, [2, 1, 0]]

            # np => tensor
            img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()[None, ...].to(self.device)
            with torch.no_grad():
                for i in range(len(img)):
                    x_np = util.tensor2img(img.detach()[0].float().cpu())

                    imscore = util.laplacian(x_np).mean()
                    pred_psnr_list = []

                    for j in range(len(self.hmap['cost_list'])):
                        pred_psnr_list.append(util.hmap_y_mean_to_pred_psnr(self.hmap['hmap_y_mean_list'][j], self.hmap['hmap_x_list'][j], imscore))

                    pred_psnr_list = np.array(pred_psnr_list)
                    delt_pred_psnr_list = pred_psnr_list


                    _type = torch.zeros((1, len(self.hmap['cost_list']))).to(img.device)

                    if self.hmap['eta'] == 'best':
                        _type[0][np.argmax(pred_psnr_list)] += 1
                    elif self.hmap['eta'] < 100:
                        _type[0][self.hmap['eta']] += 1
                    else:
                        xjb = self.hmap['eta'] * delt_pred_psnr_list - self.hmap['cost_list']
                        _type[0][np.argmax(xjb)] += 1




                    flag = torch.max(_type, 1)[1].data.squeeze()
                    p = F.softmax(_type, dim=1)

                    if flag == 0:
                        out = F.interpolate(img[i].unsqueeze(0), scale_factor=4, mode='bicubic', align_corners=False)      # 25.50 43%
                    else:
                        self.netG.apply(lambda m: setattr(m, 'width_id', flag))
                        self.netG.apply(lambda m: setattr(m, 'width_mult', self.mult_list[flag]))
                        out = self.netG(img[i].unsqueeze(0))





                    if i == 0:
                        srt = out
                        _type = p
                    else:
                        srt = torch.cat((srt, out), 0)
                        _type = torch.cat((_type, p), 0)


            sr_img = util.tensor2img(srt.detach()[0].float().cpu())
            sr_list.append(sr_img)

            if index == 0:
                type_res = _type
            else:
                type_res = torch.cat((type_res, _type), 0)


            psnr = util.calculate_psnr(sr_img, GT_img)
            flag = torch.max(_type, 1)[1].data.squeeze()

            psnr_type[flag] += psnr

            index += 1

        self.fake_H = self.combine(sr_list, num_h, num_w, h, w, self.patch_size, self.step)
        
        if self.opt['add_mask']:
            self.fake_H_mask = self.combine_addmask(sr_list, num_h, num_w, h, w, self.patch_size, self.step, type_res)

        self.real_H = self.real_H[0:h * self.scale, 0:w * self.scale, :]
        self.num_res = self.print_res(type_res)
        self.psnr_res = psnr_type
        
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals_patch(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def get_current_visuals_image(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L
        out_dict['rlt'] = self.fake_H
        out_dict['num_res'] = self.num_res
        out_dict['psnr_res']=self.psnr_res
        if need_GT:
            out_dict['GT'] = self.real_H
        if self.opt['add_mask']:
            out_dict['rlt_mask']=self.fake_H_mask
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info(f'Loading model.netG from [{load_path_G}] ...')
            self.load_network(load_path_G, self.netG)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def crop_cpu(self, img, crop_sz, step):
        n_channels = len(img.shape)
        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            h, w, c = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))
        h_space = np.arange(0, h - crop_sz + 1, step)
        w_space = np.arange(0, w - crop_sz + 1, step)
        index = 0
        num_h = 0
        lr_list=[]
        for x in h_space:
            num_h += 1
            num_w = 0
            for y in w_space:
                num_w += 1
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz]
                else:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                lr_list.append(crop_img)
        h=x + crop_sz
        w=y + crop_sz
        return lr_list, num_h, num_w, h, w

    def combine(self, sr_list, num_h, num_w, h, w, patch_size, step):
        index=0
        sr_img = np.zeros((h*self.scale, w*self.scale, 3), 'float32')
        for i in range(num_h):
            for j in range(num_w):
                sr_img[i*step*self.scale:i*step*self.scale+patch_size*self.scale,j*step*self.scale:j*step*self.scale+patch_size*self.scale,:]+=sr_list[index]
                index+=1
        sr_img=sr_img.astype('float32')

        for j in range(1,num_w):
            sr_img[:,j*step*self.scale:j*step*self.scale+(patch_size-step)*self.scale,:]/=2

        for i in range(1,num_h):
            sr_img[i*step*self.scale:i*step*self.scale+(patch_size-step)*self.scale,:,:]/=2
        return sr_img

    def combine_addmask(self, sr_list, num_h, num_w, h, w, patch_size, step, _type):
        index = 0
        sr_img = np.zeros((h * self.scale, w * self.scale, 3), 'float32')

        for i in range(num_h):
            for j in range(num_w):
                sr_img[i * step * self.scale:i * step * self.scale + patch_size * self.scale,
                j * step * self.scale:j * step * self.scale + patch_size * self.scale, :] += sr_list[index]
                index += 1
        sr_img = sr_img.astype('float32')

        for j in range(1, num_w):
            sr_img[:, j * step * self.scale:j * step * self.scale + (patch_size - step) * self.scale, :] /= 2

        for i in range(1, num_h):
            sr_img[i * step * self.scale:i * step * self.scale + (patch_size - step) * self.scale, :, :] /= 2

        index2 = 0
        for i in range(num_h):
            for j in range(num_w):
                # add_mask
                alpha = 1
                beta = 0.2
                gamma = 0
                bbox1 = [j * step * self.scale + 8, i * step * self.scale + 8,
                         j * step * self.scale + patch_size * self.scale - 9,
                         i * step * self.scale + patch_size * self.scale - 9]  # xl,yl,xr,yr
                zeros1 = np.zeros((sr_img.shape), 'float32')

                if torch.max(_type, 1)[1].data.squeeze()[index2] == 0:
                    mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=(0, 255, 0), thickness=-1)# simple green
                elif torch.max(_type, 1)[1].data.squeeze()[index2] == 1:
                    mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=(0, 255, 255), thickness=-1)# medium yellow
                elif torch.max(_type, 1)[1].data.squeeze()[index2] == 2:
                    mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=(0, 0, 255), thickness=-1)# hard red

                sr_img = cv2.addWeighted(sr_img, alpha, mask2, beta, gamma)
                index2+=1
        return sr_img

    def print_res(self, type_res):
        num = [0] * len(self.opt['network_G']['width_list'])
        for i in torch.max(type_res, 1)[1].data.squeeze():
            num[i] += 1
        return num


