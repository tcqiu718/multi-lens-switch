import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
# from model.IFNet_m import *
from model.AMT import Model as AMTModel
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
from losses.vgg import VGGLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_timestep(timestep, reference):
    if torch.is_tensor(timestep):
        embt = timestep.to(device=reference.device, dtype=reference.dtype)
    else:
        embt = torch.tensor(timestep, device=reference.device, dtype=reference.dtype)

    batch_size = reference.shape[0]
    if embt.numel() == 1:
        return embt.reshape(1, 1, 1, 1).expand(batch_size, 1, 1, 1)
    if embt.numel() == batch_size:
        return embt.reshape(batch_size, 1, 1, 1)
    raise ValueError(
        "Expected one timestep or one timestep per sample, got {} values for batch size {}".format(
            embt.numel(), batch_size
        )
    )


class Model:
    def __init__(self, local_rank=-1, ):
        self.flownet = AMTModel()  
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3)
        self.epe = EPE()
        self.lap = LapLoss() 
        self.sobel = SOBEL()
        self.vgg = VGGLoss().cuda()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_pretrained_model(self, path, rank=0, suffix=None, convert=False):
        def convert(param):
            return {"module." + k: v
                for k, v in param.items()
                
            }
            
        if suffix is None:
            load_path = '{}/flownet.pth'.format(path)
        else:
            load_path = '{}/{}_flownet.pth'.format(path, suffix)

        if convert:
            self.flownet.load_state_dict(convert(torch.load(load_path)['state_dict']))
            return None

        if rank <0:
            self.flownet.load_state_dict(convert(torch.load(load_path)['state_dict']))
        else:
            self.flownet.load_state_dict(torch.load(load_path)['state_dict'])

        
    def load_model(self, path, rank=0, suffix=None):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if suffix is None:
            load_path = '{}/flownet.pkl'.format(path)
        else:
            load_path = '{}/{}_flownet.pkl'.format(path, suffix)

        if rank <0:
            self.flownet.load_state_dict(convert(torch.load(load_path)))
        else:
           self.flownet.load_state_dict(torch.load(load_path))
        
    def save_model(self, path, rank=0, suffix=None):
        if rank == 0:
            if suffix is None:
                torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))
            else:
                torch.save(self.flownet.state_dict(),'{}/{}_flownet.pkl'.format(path, suffix))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale

        embt = prepare_timestep(timestep, img0)
        pred = self.flownet(img0, img1, embt=embt)
       
        return pred

    def update(self, imgs, gt, timestep=0.5, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]

        if training:
            self.train()
        else:
            self.eval()

        embt = prepare_timestep(timestep, img0)
        pred = self.flownet(img0, img1, embt=embt)

        loss_l1 = (self.lap(pred, gt)).mean()
        loss_vgg = (self.vgg(pred, gt)).mean()

        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_vgg
            loss_G.backward()
            self.optimG.step()

        return pred, {'loss_l1':loss_l1, 'loss_vgg':loss_vgg}
