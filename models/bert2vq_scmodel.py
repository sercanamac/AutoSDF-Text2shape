
import os
from collections import OrderedDict


import numpy as np
import einops
import mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

# renderer
import pytorch3d

import torchvision.utils as vutils
import torchvision.transforms as transforms

from torch.hub import load_state_dict_from_url

from models.base_model import BaseModel
from models.networks.bert2vq_sc import BERT2VQ
from models.networks.pvqvae_networks.auto_encoder import PVQVAE
import utils
from utils.util import NoamLR
# from utils.util_3d import init_mesh_renderer, render_mesh, init_snet_to_pix3dvox_params, render_sdf, snet_to_pix3dvox

class BERT2VQSCModel(BaseModel):
    def name(self):
        return 'BERT2VQSC-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()

        # -------------------------------
        # Define Networks
        # -------------------------------

        
        #bert_conf = OmegaConf.load(opt.bert_cfg)
 
        
        # init resnet2vq network
        self.net = BERT2VQ(opt)
        self.net.to(opt.device)

        if self.isTrain:
            # ----------------------------------
            # define loss functions
            # ----------------------------------
            #self.criterion_nce = nn.CrossEntropyLoss()
            self.criterion_nce = nn.MSELoss()
            self.criterion_nce.to(opt.device)

            # ---------------------------------
            # initialize optimizers
            # ---------------------------------
            self.optimizer = optim.AdamW([p for p in self.net.parameters() if p.requires_grad == True], lr=opt.lr)

            self.scheduler = NoamLR(self.optimizer, warmup_steps=10)
            
            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)
    
    def set_input(self, input, gen_order=None):
        self.z_prev = input["z_set_prev"]
        self.z_target = input["z_set_target"]
        #self.z_shape = self.z1.shape
        

        self.text = input['current_text']


        vars_list = ['z_prev']

        self.tocuda(var_names=vars_list)
    
    def inference(self, data, should_render=True, verbose=False):
        pass

    def forward(self):
        self.outp = self.net(self.text, self.z_prev)
    
    def backward(self):
        '''backward pass for the Lang to (P)VQ-VAE code model'''
        target = self.z_target.to(self.outp.device)
        outp = self.outp
        target = rearrange(target, 'bs d1 d2 d3 c -> bs c d1 d2 d3')
       
        loss_nll = self.criterion_nce(outp, target)

        self.loss = loss_nll

        self.loss.backward()
    
    def optimize_parameters(self, total_steps):
        # self.vqvae.train()

        self.set_requires_grad([self.net], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.total_steps = total_steps
        
    
    def get_current_errors(self):
        
        ret = OrderedDict([
            ('nll', self.loss.data),
        ])

        return ret
    
    def get_current_visuals(self):
                            
        return OrderedDict()
    
    def eval_metrics(self, dataloader, thres=0.0):
                            
        return OrderedDict()

    def save(self, label, epoch=None):

        state_dict = {
            # 'vqvae': self.vqvae.cpu().state_dict(),
            'bert2vq': self.net.cpu().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'total_steps': self.total_steps
        }
    
        save_filename = 'bert2vq_%s.pth' % (label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(state_dict, save_path)
        # self.vqvae.to(self.opt.device)
        self.net.to(self.opt.device)

    def load_ckpt(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt

        # self.vqvae.load_state_dict(state_dict['vqvae'])
        self.net.load_state_dict(state_dict['bert2vq'])
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])
            
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))