import torch
from collections import OrderedDict
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.archs import build_network
from basicsr.models.sr_model import SRModel
from basicsr.models.video_recurrent_model import VideoRecurrentModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class RecurrentMixPrecisionRTModel(VideoRecurrentModel):
    """VRT Model adopted in the original VRT. Mix precision is adopted.

    Paper: A Video Restoration Transformer
    """

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.net_g.to(self.device)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
            self.fix_flow_iter = opt['train'].get('fix_flow')

    # add use_static_graph
    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
            use_static_graph = self.opt.get('use_static_graph', False)
            if use_static_graph:
                logger = get_root_logger()
                logger.info(
                    f'Using static graph. Make sure that "unused parameters" will not change during training loop.')
                net._set_static_graph()
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                # add 'deform'
                if 'spynet' in name or 'deform' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])

        # # adopt mix precision
        # use_apex_amp = self.opt.get('apex_amp', False)
        # if use_apex_amp:
        #     self.net_g, self.optimizer_g = apex_amp_initialize(
        #         self.net_g, self.optimizer_g, init_args=dict(opt_level='O1'))
        #     logger = get_root_logger()
        #     logger.info(f'Using apex mix precision to accelerate.')

        # adopt DDP
        self.net_g = self.model_to_device(self.net_g)
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, scaler, current_iter):

        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'deform' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        # update the gradient when forward 4 times
        self.optimizer_g.zero_grad()

        with autocast():
            self.output = self.net_g(self.lq)
            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix
            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style
            scaler.scale(l_total).backward()
            scaler.step(self.optimizer_g)
            scaler.update()

            # l_total.backward()
            # self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


