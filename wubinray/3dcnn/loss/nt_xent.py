import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class NTXentLoss(nn.Module):

    def __init__(self, device, bsize, temperature):
        super(NTXentLoss, self).__init__()

        self.device = device

        self.bsize = bsize
        self.temperature = temperature 

        self.similarity_F = nn.CosineSimilarity(dim=2)
        self.mask_same_inst = self.get_mask_same_inst()
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, zis, zjs):
        zs = torch.cat([zis, zjs], dim=0)

        sim = self.similarity_F(zs.unsqueeze(1), 
                                zs.unsqueeze(0)) / self.temperature #(2N,2N)
       
        scalar = sim.clone()
        #scalar = torch.abs(scalar)
        #scalar = 1.0 + F.tanh(scalar/2.0) #tanh: 77.0125
        #scalar = F.relu(scalar+1.0)       #relu: 77.2125
        #scalar = F.sigmoid(scalar/1.0)*2  #sigmoid: 77.775
        #sim = sim * scalar.detach()

        sim_ij = torch.diag(sim, self.bsize) #(N)
        sim_ji = torch.diag(sim, -self.bsize) #(N)

        M = 2*self.bsize 

        # pos:(2N), neg:(2N,2N-2) with_mask_same_inst
        positive_samples = torch.cat([sim_ij, sim_ji], dim=0).reshape(M, 1)
        negative_samples = sim[self.mask_same_inst].reshape(M, M-2)
        
        #######
        #scaler = negative_samples.clone()
        #print(scaler.max(), scaler.min())
        #scaler = .0 + F.tanh(scaler/2.0)
        #scaler.requires_grad=False
        #negative_samples *= scaler.detach()
        #######

        logits = torch.cat([positive_samples, negative_samples], dim=-1)
        labels = torch.zeros(M).to(positive_samples.device).long()
        
        loss = self.ce_loss(logits, labels)
        return loss 
    
    def get_mask_same_inst(self):
        diag = np.eye(2*self.bsize)
        up_triangle = np.eye(2*self.bsize, k=self.bsize)
        lo_triangle = np.eye(2*self.bsize, k=-self.bsize)
        mask = torch.from_numpy((diag + up_triangle + lo_triangle))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device) 

