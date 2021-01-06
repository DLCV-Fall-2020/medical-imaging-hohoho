import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature 
        pass

    def forward(self, features, labels, mask=None):
        '''
        Args:
            features: (b, z)
            labels: (b, cls)
            mask: (b, 1)
        Returns:
            loss scalar
        Ref: 
            https://github.com/HobbitLong/SupContrast/blob/master/losses.py
        '''
        # mask, remove null channels
        bsize = mask.sum().item()
        features = torch.masked_select(features, mask.bool()).view(bsize,-1)
        labels = torch.masked_select(labels, mask.bool()).view(bsize,-1)
        labels = labels.float()
        
        assert len(features) == len(labels)
 
        # dot samples (N,N)
        similarity = F.cosine_similarity(features.unsqueeze(1), 
                                         features.unsqueeze(0), dim=2) / self.temperature
        similarity_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - similarity_max.detach()

        exp_logits = torch.exp(similarity) # (N,N)
        
        # positive samples, mutual info
        mask_pos = torch.mm(labels, labels.T).float()
        mask_pos[np.diag_indices(bsize)] = 0 
        pos = (similarity * mask_pos.detach()).sum(1, keepdim=True)

        # contrast samples
        mask_neg = (mask_pos==0) + mask_pos
        mask_neg[np.diag_indices(bsize)] = 0
        log_exp_neg = torch.log((exp_logits * mask_neg.detach()).sum(1, keepdim=True) + 1e-10)
        
        # NCE loss
        loss = - (pos - log_exp_neg)
        return loss.mean()
        
