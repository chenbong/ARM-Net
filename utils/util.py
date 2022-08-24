import os
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
import pandas as pd
import torch.nn as nn
from torchvision.utils import make_grid
import warnings
from scipy.special import softmax
import matplotlib.pyplot as plt
from scipy import stats

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper



def getnearpos(array, value): return (np.abs(array-value)).argmin()

def get_netG(model):
    if isinstance(model.netG, nn.DataParallel):
        return model.netG.module
    else:
        return model.netG

def get_net(model):
    if isinstance(model.netG, nn.DataParallel):
        return model
    else:
        return model.netG


def save_hmap(hmap, hmap_x, hmap_y, save_path, bins):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.matshow(hmap, origin='lower')
    ax.xaxis.set_ticks(np.arange(0,bins,1))
    ax.set_xticklabels(np.around(np.linspace(min(hmap_x), max(hmap_x), num=bins), 2), rotation=90)
    ax.xaxis.set_ticks_position('bottom')

    ax.yaxis.set_ticks(np.arange(0,bins,1))
    ax.set_yticklabels(np.around(np.linspace(min(hmap_y), max(hmap_y), num=bins), 2))

    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.05)

    plt.cla()
    plt.close("all")

def read_log(log_path):
    nums = []
    with open(log_path, 'r') as f:
        # print(f.read())    # 
        for line in f.readlines():    # 
            num = eval(line.strip().split(', ')[-1])
            nums.append(num)
    return nums

def calc_base_hmap(imscore_log_path, psnr_log_path, bins):
    x = read_log(imscore_log_path)
    y = read_log(psnr_log_path)
    z = [1] * len(x)

    fig, ax = plt.subplots()
    hmap, hmap_x, hmap_y, _ =  ax.hist2d(x, y, bins=bins)

    ret = stats.binned_statistic(x, y, 'mean', bins=hmap_x)
    hmap_y_mean = ret.statistic

    hmap_x = hmap_x[:-1] + (hmap_x[1]-hmap_x[0])/2
    hmap_y = hmap_y[:-1] + (hmap_y[1]-hmap_y[0])/2

    plt.cla()
    plt.close("all")

    return x, y, hmap, hmap_x, hmap_y, hmap_y_mean


def hmap_imscore_to_pred_psnr(hmap, hmap_x, hmap_y, imscore, t=1e-2):
    idx = getnearpos(hmap_x, imscore)
    psnr_dist = softmax(hmap[:,idx] / t)
    pred_psnr = random.choices(hmap_y, weights=psnr_dist)[0]
    return pred_psnr

def hmap_y_mean_to_pred_psnr(hmap_y_mean, hmap_x, imscore):
    idx = getnearpos(hmap_x, imscore)
    pred_psnr = hmap_y_mean[idx]
    return pred_psnr

layer_modules = (
    nn.Conv2d, nn.ConvTranspose2d,
    nn.Linear,
    nn.BatchNorm2d,
)

def summary(model, input_size, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    x = torch.zeros(input_size).to(next(model.parameters()).device)

    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            key = None
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)
                    break
            assert key

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["macs"] = 0, 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                        assert len(inputs[0].size()) == 4 and len(inputs[0].size()) == len(outputs[0].size())+1

                        in_c, in_h, in_w = inputs[0].size()[1:]
                        k_h, k_w = module.kernel_size
                        out_c, out_h, out_w = outputs[0].size()
                        groups = module.groups
                        kernel_mul = k_h * k_w * (in_c // groups)

                        kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
                        total_mul = kernel_mul_group * groups
                        info["macs"] += 2 * total_mul


                    elif isinstance(module, nn.BatchNorm2d):
                        info["macs"] += inputs[0].size()[1]
                    else:
                        info["macs"] += param.nelement()

                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        if isinstance(module, layer_modules) or not module._modules:
            hooks.append(module.register_forward_hook(hook))



    module_names = get_names_dict(model)

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    try:
        with torch.no_grad():
            model(x) if not (kwargs or args) else model(x, *args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()
    # Use pandas to align the columns
    df = pd.DataFrame(summary).T


    df["Mult-Adds"] = pd.to_numeric(df["macs"], errors="coerce")
    df["Params"] = pd.to_numeric(df["params"], errors="coerce")
    df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
    df = df.rename(columns=dict(
        ksize="Kernel Shape",
        out="Output Shape",
    ))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        df_sum = df.sum()


    df.index.name = "Layer"

    df = df[["Kernel Shape", "Output Shape", "Params", "Mult-Adds"]]
    max_repr_width = max([len(row) for row in df.to_string().split("\n")])

    return df_sum["Mult-Adds"]

def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_"+ key if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names


def laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

############################################

def cal_FLOPs(num_ress, cost_list):
    assert len(num_ress) == len(cost_list)
    flops = 0
    for i in range(len(cost_list)):
        flops += cost_list[i] * num_ress[i]
    flops /= sum(num_ress)
    percent = flops / cost_list[-1]

    return flops, percent




def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        print(log_file)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################
def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    #**
    return
    # cv2.imwrite(img_path, img)



####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float(80)
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()
