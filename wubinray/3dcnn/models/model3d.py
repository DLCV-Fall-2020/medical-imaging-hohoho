#from models.resnet import *
from models.resnet3d import * 

import torch.nn as nn
import torch 

__all__=["HemoModel3d"]

class HemoModel3d(nn.Module):
    def __init__(self, backbone="resnet10_3d", in_channels=1, hidd_dim=128, proj_dim=32, n_classes=5):
        super().__init__()

        # backbone
        kwargs = dict(in_channels=in_channels, hidd_dim=hidd_dim)
        if backbone=="resnet10_3d":
            self.backbone = resnet10_3d(**kwargs)
        elif backbone=="resnet18_3d":
            self.backbone = resnet18_3d(**kwargs)

        # proj
        self.projector = nn.Sequential(
                    #nn.Linear(hidd_dim, hidd_dim),
                    #nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(hidd_dim, proj_dim)
                )
    
    def forward(self, x):
        '''
        Args:
            x:(b,t,h,w)
        Return:
            x:(b,t,z)
        '''
        b,ch,t,_,_ = x.shape

        h = self.backbone(x)
        z = self.projector(h)
        return h, z

