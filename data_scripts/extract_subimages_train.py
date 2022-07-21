"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.util import ProgressBar  # noqa: E402
import data.util as data_util  # noqa: E402


#Then we densely crop 1.59M sub-images with size 32 × 32 from LR images.


def main():
    mode = 'pair'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
    opt = {}
    opt['n_thread'] = 16
    opt['compression_level'] = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.
    if mode == 'pair':
        # cut training data
        GT_folder = '/media/DATA2/SR/DIV2K/TMP/DIV2K_train_HR_scale/HR/x4'# fix to your path
        LR_folder = '/media/DATA2/SR/DIV2K/TMP/DIV2K_train_HR_scale/LR/x4'# fix to your path
        save_GT_folder = '/media/DATA2/SR/DIV2K/TMP/DIV2K_scale_sub/GT'
        save_LR_folder = '/media/DATA2/SR/DIV2K/TMP/DIV2K_scale_sub/LR'

        scale_ratio = 4
        crop_sz = 128  # the size of each sub-image (GT)
        step = 64  # step of the sliding crop window (GT)
        thres_sz = 48  # size threshold
        ########################################################################
        # check that all the GT and LR images have correct scale ratio
        img_GT_list = data_util._get_paths_from_images(GT_folder)
        img_LR_list = data_util._get_paths_from_images(LR_folder)
        assert len(img_GT_list) == len(img_LR_list), 'different length of GT_folder and LR_folder.'
        for idx, (path_GT, path_LR) in enumerate(zip(img_GT_list, img_LR_list)):
            print(f'{idx}, {path_GT}, {path_LR}')
            # /media/DATA2/SR/DIV2K/TMP/DIV2K_train_HR_scale/LR/x4/0.8_0682.png
            # /media/DATA2/SR/DIV2K/TMP/DIV2K_train_HR_scale/GT/x4/0.8_0682.png

            img_GT = Image.open(path_GT)
            img_LR = Image.open(path_LR)
            w_GT, h_GT = img_GT.size
            w_LR, h_LR = img_LR.size
            assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR width [{:d}] for {:s}.'.format(w_GT, scale_ratio, w_LR, path_GT)
            assert h_GT / h_LR == scale_ratio, 'GT height [{:d}] is not {:d}X as LR height [{:d}] for {:s}.'.format(h_GT, scale_ratio, h_LR, path_GT)
        # check crop size, step and threshold size
        assert crop_sz % scale_ratio == 0, 'crop size is not {:d}X multiplication.'.format(scale_ratio)
        assert step % scale_ratio == 0, 'step is not {:d}X multiplication.'.format(scale_ratio)
        assert thres_sz % scale_ratio == 0, 'thres_sz is not {:d}X multiplication.'.format(scale_ratio)
        
        print('process GT...')
        opt['input_folder'] = GT_folder
        opt['save_folder'] = save_GT_folder
        opt['crop_sz'] = crop_sz
        opt['step'] = step
        opt['thres_sz'] = thres_sz
        extract_signle(opt)
        print('process LR...')
        opt['input_folder'] = LR_folder
        opt['save_folder'] = save_LR_folder
        opt['crop_sz'] = crop_sz // scale_ratio
        opt['step'] = step // scale_ratio
        opt['thres_sz'] = thres_sz // scale_ratio
        extract_signle(opt)
        assert len(data_util._get_paths_from_images(save_GT_folder)) == len(
            data_util._get_paths_from_images(
                save_LR_folder)), 'different length of save_GT_folder and save_LR_folder.'
    elif mode == 'single':
        # opt['input_folder'] = '/media/DATA2/SR/DIV2K/TMPDIV2K_HR/DIV2K_valid_HR'
        opt['input_folder'] = '/media/DATA2/SR/DIV2K/DIV2K_valid_HR'

        # opt['save_folder'] = '/media/DATA2/SR/DIV2K/TMPDIV2K_valid_sub/GT'
        opt['save_folder'] = '/media/DATA2/SR/DIV2K/TMP/DIV2K_valid_sub/GT'

        opt['crop_sz'] = 256  # the size of each sub-image
        opt['step'] = 256  # step of the sliding crop window
        opt['thres_sz'] = 48  # size threshold
        extract_signle(opt)
    else:
        raise ValueError('Wrong mode.')


def extract_signle(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)
    img_list = data_util._get_paths_from_images(input_folder)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, opt):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         img_name.replace('.png', '_s{:03d}.png'.format(index))), crop_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
