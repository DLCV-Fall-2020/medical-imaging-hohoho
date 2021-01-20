import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class HemoSupConLoss(nn.Module):
    def __init__(self, pos_weight=12, temperature=0.5):
        super().__init__()
        self.pos_weight = pos_weight 
        self.temperature = temperature 
        pass

    def forward(self, features, labels):
        '''
        Args:
            features: (b, z)
            labels: (b, cls)
        Returns:
            loss scalar
        Ref: 
            https://github.com/HobbitLong/SupContrast/blob/master/losses.py
        '''
        bsize = features.size(0)
        assert len(labels) == bsize
        labels = labels.float()
 
        # dot samples (N,N)
        similarity = F.cosine_similarity(features.unsqueeze(1), 
                                         features.unsqueeze(0), dim=2) / self.temperature
        similarity_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - similarity_max.detach()

        exp_logits = torch.exp(similarity) # (N,N)
        
        # positive samples, mutual info, have blood
        mask = torch.mm(labels, labels.T).float()
        mask[np.diag_indices(bsize)] = 0 
        mask *= self.pos_weight # have blood's pos weight
        pos = (similarity * mask.detach()).sum(1, keepdim=True)

        # positive samples, mutual info, no blood
        mask = torch.mm((1-labels), (1-labels).T).float() / (5*5)
        mask[np.diag_indices(bsize)] = 0
        pos += (similarity * mask.detach()).sum(1, keepdim=True)

        # contrast samples
        mask_neg = (mask==0) + mask
        mask_neg[np.diag_indices(bsize)] = 0
        log_exp_neg = torch.log((exp_logits * mask_neg.detach()).sum(1, keepdim=True) + 1e-10)
        
        # NCE loss
        loss = - (pos - log_exp_neg)
        return loss.mean()
        
