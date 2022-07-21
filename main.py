import os
import math
import argparse
import random
import logging
import time
import numpy as np
import torch
import datetime
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy import stats

import options.options as option
from utils import util
from data.util import bgr2ycbcr
from data import create_dataloader, create_dataset
from models import create_model
from utils.util import summary


abspath = os.path.abspath(__file__)


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, help='Path to option YAML file.')
parser.add_argument('-root', type=str, default=None, choices=['.'])
args = parser.parse_args()
opt = option.parse(args.opt, root=args.root)

if args.root is not None:
    if opt['path']['pretrain_model_G'] is None and os.path.isfile(os.path.join(args.root, 'exp/models/latest.pth')):
        opt['path']['pretrain_model_G'] = 'exp/models/latest.pth'
    if opt['path']['resume_state'] is None and os.path.isfile(os.path.join(args.root, 'exp/training_state/latest.state')):
        opt['path']['resume_state'] = 'exp/training_state/latest.state'


opt_net = opt['network_G']
which_model = opt_net['which_model_G']



if not os.path.isdir(opt['path']['job_dir']):
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'job_dir' and 'pretrain_model' not in key and 'resume' not in key))


util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))



opt = option.dict_to_nonedict(opt)




test_loaders = []
val_img_loaders = []

for phase, dataset_opt in opt['datasets'].items():
    if phase == 'train':
        train_set = create_dataset(dataset_opt)
        train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
        total_iters = int(opt['train']['niter'])
        total_epochs = int(math.ceil(total_iters / train_size))
        train_sampler = None
        train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))


    elif 'val_patch' in phase:
        val_patch_set = create_dataset(dataset_opt)
        val_patch_loader = create_dataloader(val_patch_set, dataset_opt, opt, None)
        logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_patch_set)))
    elif 'val_img' in phase:
        val_img_set = create_dataset(dataset_opt)
        val_img_loader = create_dataloader(val_img_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_img_set)))
        val_img_loaders.append(val_img_loader)
    elif 'test' in phase:
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)
    else:
        raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
assert train_loader is not None

model = create_model(opt)


width_list = np.array(opt['network_G']['width_list'])
nf = np.array(opt['network_G']['nf'])
mult_list = width_list / nf



cost_list = [0.0]
for width_id, mult in enumerate(mult_list):
    if mult == 0: continue
    model.netG.apply(lambda m: setattr(m, 'width_id', width_id))
    model.netG.apply(lambda m: setattr(m, 'width_mult', mult_list[width_id]))

    cost_list.append(summary(model.netG, (1, 3, 32, 32)))


logger.info(f'--- cost_list:{cost_list} ---')


netG = util.get_netG(model)


if opt['path']['resume_state']:
    device_id = torch.cuda.current_device()
    resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
    logger.info(f"Loading state from: {opt['path']['resume_state']}, epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")

    start_epoch = resume_state['epoch']
    current_step = resume_state['iter']
    model.resume_training(resume_state)
else:
    model.hmap['cost_list'] = np.array(cost_list)
    model.hmap['eta'] = 'best'
    model.hmap['bins'] = 30

    start_epoch = 0
    current_step = 0


if opt['path']['pretrain_model_G']:
    model.load()


def train():
    global start_epoch, current_step

    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    iter_time = util.AverageMeter('1w Iter Time:', ':.2f')
    start = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            if not os.path.exists(abspath):
                exit()
            current_step += 1
            if current_step > total_iters:
                break
            model.optimizer_G.zero_grad()

            width_id = random.choices(list(range(1, len(opt['network_G']['width_list']))), weights=model.hmap['cost_list'][1:]**2)[0]

            model.netG.apply(lambda m: setattr(m, 'width_id', width_id))
            model.netG.apply(lambda m: setattr(m, 'width_mult', mult_list[width_id]))

            model.feed_data(train_data)

            model.fake_H = model.netG(model.var_L)
            l_pix = model.cri_pix(model.fake_H, model.real_H)
            loss = l_pix
            loss.backward()

            model.optimizer_G.step()

            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            model.log_dict['l_pix'] = l_pix.item()

            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)


            if (opt['datasets'].get('val_patch', None) or opt['datasets'].get('val_img', None)) and current_step % opt['train']['val_freq'] == 0:
                test_image(test_loaders=val_img_loaders)

            if current_step % opt['logger']['save_checkpoint_freq'] == 0 or current_step > opt['train']['niter']-10:
                model.save(current_step)
                model.save_training_state(epoch, current_step)

            if current_step % 10000 == 0:
                iter_time.update(time.time() - start)
                start = time.time()
                finish_time = time.time() + iter_time.avg * (total_iters-1 - current_step) / 1e4
                finish_dt = datetime.datetime.fromtimestamp(finish_time).strftime("%m/%d %H:%M:%S")
                logger.info(f"======> {iter_time}s, Will finish at: {finish_dt}")

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')



def test_patch(test_patch_loaders, width_id, width_mult, current_step):
    model.netG.apply(lambda m: setattr(m, 'width_id', width_id))
    model.netG.apply(lambda m: setattr(m, 'width_mult', width_mult))

    patch_imscores = []
    patch_psnrs = []
    for test_patch_loader in test_patch_loaders:
        avg_psnr = 0.
        idx = 0
        model.netG.eval()
        for test_data in test_patch_loader:

            idx += 1
            img_name = os.path.splitext(os.path.basename(test_data['LQ_path'][0]))[0]
            img_dir = os.path.join(opt['path']['val_images'], img_name)
            model.feed_data(test_data)

            lr_img = util.tensor2img(test_data['LQ'].detach()[0].float().cpu())

            imscore = util.laplacian(lr_img).mean()
            patch_imscores.append(imscore)
            model.test_patch()

            visuals = model.get_current_visuals_patch()

            sr_img = util.tensor2img(visuals['rlt'])
            gt_img = util.tensor2img(visuals['GT'])

            sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
            psnr = util.calculate_psnr(sr_img, gt_img)
            patch_psnrs.append(psnr)

            avg_psnr += psnr
        model.netG.train()

        avg_psnr = avg_psnr / idx

        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    return patch_imscores, patch_psnrs


def test_image(test_loaders, is_calc_hmap=True):
    if is_calc_hmap:
        for width_id, mult in enumerate(mult_list):
            if mult == 0:
                x, y, hmap, hmap_x, hmap_y, hmap_y_mean = util.calc_base_hmap('utils/bilinear_imscore.log', 'utils/bilinear_psnr.log', bins=model.hmap['bins'])

                model.hmap['hmap_list'].append(hmap)
                model.hmap['hmap_x_list'].append(hmap_x)
                model.hmap['hmap_y_list'].append(hmap_y)
                model.hmap['hmap_y_mean_list'].append(hmap_y_mean)
                util.save_hmap(hmap, hmap_x, hmap_y, os.path.join('utils', 'hmap_0.png'), bins=model.hmap['bins'])
            else:
                model.netG.apply(lambda m: setattr(m, 'width_id', width_id))
                model.netG.apply(lambda m: setattr(m, 'width_mult', mult_list[width_id]))
                patch_imscores, patch_psnrs = test_patch(test_patch_loaders=[val_patch_loader], width_id=width_id, width_mult=mult_list[width_id], current_step=-1)

                fig, ax = plt.subplots(figsize=(6.4, 6.4))
                hmap, hmap_x, hmap_y, _ =  ax.hist2d(patch_imscores, patch_psnrs, bins=model.hmap['bins'])
                ret = stats.binned_statistic(patch_imscores, patch_psnrs, 'mean', bins=hmap_x)
                hmap_y_mean = ret.statistic

                hmap_x = hmap_x[:-1] + (hmap_x[1]-hmap_x[0])/2
                hmap_y = hmap_y[:-1] + (hmap_y[1]-hmap_y[0])/2

                model.hmap['hmap_list'].append(hmap)
                model.hmap['hmap_x_list'].append(hmap_x)
                model.hmap['hmap_y_list'].append(hmap_y)
                model.hmap['hmap_y_mean_list'].append(hmap_y_mean)

                util.save_hmap(hmap, hmap_x, hmap_y, os.path.join('utils', f'hmap_{width_id}.png'), bins=model.hmap['bins'])

                plt.cla()
                plt.close("all")


    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        dataset_dir = os.path.join(opt['path']['job_dir'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        num_ress = [0] * (len(opt['network_G']['width_list']))


        for data in test_loader:
            if not os.path.exists(abspath):
                exit()
            need_GT = True
            model.feed_data(data, need_GT=need_GT)
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            model.test_image()
            visuals = model.get_current_visuals_image(need_GT=need_GT)

            sr_img = visuals['rlt']
            if opt['add_mask']:
                sr_img_mask=visuals['rlt_mask']

            num_res = visuals['num_res']
            psnr_res = visuals['psnr_res']

            suffix = opt['suffix']
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = os.path.join(dataset_dir, img_name + '.png')
            util.save_img(sr_img, save_img_path)
            if opt['add_mask']:
                util.save_img(sr_img_mask, save_img_path.split('.pn')[0]+'_mask.png')


            if need_GT:
                gt_img = visuals['GT']
                sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])

                psnr = util.calculate_psnr(sr_img, gt_img)

                test_results['psnr'].append(psnr)

                if gt_img.shape[2] == 3:
                    sr_img_y = bgr2ycbcr(sr_img / 255., only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img / 255., only_y=True)
                    psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)

                    test_results['psnr_y'].append(psnr_y)
                    for i in range(len(opt['network_G']['width_list'])):
                        num_ress[i] += num_res[i]

                    flops, percent = util.cal_FLOPs(num_res, cost_list)
                    logger.info(f'{img_name} - PSNR: {psnr:.4f}dB  FLOPs: {flops/1e6:.2f}M  Percent: {percent*100:.2f}%')

                else:
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr))

            else:
                logger.info(img_name)

        logger.info(f'# Validation # Class num: {num_ress}  all:{sum(num_ress)}')

        if need_GT:
            flops, percent = util.cal_FLOPs(num_ress, cost_list)
            logger.info('# FLOPs {:.4e} Percent {:.4e}'.format(flops,percent))
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])

            logger.info(f'----Average PSNR results for {test_set_name}----\tPSNR: {ave_psnr:.6f} dB\n')



if __name__ == '__main__':
    if opt['is_train']:
        train()
        logger.info(f"\n--- Test best eta ---")
        test_image(test_loaders=test_loaders, is_calc_hmap=True)
        model.save_training_state(total_epochs, current_step)

    if opt['is_test']:
        for subnet_id in range(1, len(opt['network_G']['width_list'])):
            model.hmap['eta'] = subnet_id
            logger.info(f"\n--- Test subnet_id:{subnet_id}, width:{opt['network_G']['width_list'][subnet_id]}, mult:{mult_list[subnet_id]:.2f} ---")
            test_image(test_loaders=test_loaders, is_calc_hmap = len(model.hmap['hmap_list'])==0 )

        for k in [0.5, 1, 2, 4, 8, 16, 32]:
            model.hmap['eta'] = k * model.hmap['cost_list'][-1]
            logger.info(f"\n--- Test different eta:{model.hmap['eta']} = {k}x{model.hmap['cost_list'][-1]} ---")
            test_image(test_loaders=test_loaders, is_calc_hmap= len(model.hmap['hmap_list'])==0 )



