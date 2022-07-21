import models.archs.FSRCNN_arch as FSRCNN_arch
import models.archs.CARN_arch as CARN_arch
import models.archs.SRResNet_arch as SRResNet_arch

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'fsrcnn':
        netG = FSRCNN_arch.US_FSRCNN_net(input_channels=opt_net['in_nc'],upscale=opt_net['scale'], nf=opt_net['nf'], s=opt_net['s'], m=opt_net['m'])
    elif which_model == 'carn':
        netG = CARN_arch.US_CARN_M(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], scale=opt_net['scale'], group=opt_net['group'])
    elif which_model == 'srresnet':
        netG = SRResNet_arch.US_MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG