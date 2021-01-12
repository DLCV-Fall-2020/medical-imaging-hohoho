from models.densenet import HemoDenseNet121

import torchvision.models as models 
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''
Ref: https://github.com/darraghdog/rsna/blob/master/scripts/trainlstm.py

https://reurl.cc/Ag1egj
'''
resnets = {"resnet18": models.resnet18,
           "resnet34": models.resnet34,
           "resnet50": models.resnet50}
class HemoCnnLstm(nn.Module):
    def __init__(self, backbone="resnet18", in_channels=10, n_classes=5, 
                    pretrained=None):
        super().__init__()

        hidden_dim=256
        rnncell_dim=64
        
        if "resnet" in backbone:  
            resnet = resnets[backbone](pretrained=True)
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                        stride=2, padding=3, bias=False)
            self.cnn_net = resnet 
            in_features = resnet.fc.in_features 
        elif "densnet121" is backbone:
            densnet = HemoDenseNet121(1, n_classes)
            densnet.load_state_dict(
                            torch.load(pretrained,map_location="cpu"))
            self.cnn_net = densnet.base_model 
            
            in_features = densnet.base_model.classifier.in_features 

        self.backbone = nn.Sequential(*list(self.cnn_net.children())[:-1])
        self.backbone = nn.DataParallel(self.backbone)
        
        self.project = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
            )

        self.rnn = nn.GRU(hidden_dim, rnncell_dim,
                            bidirectional=True, batch_first=True)
        #self.rnn = nn.DataParallel(self.lstm)

        self.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.ReLU(True),
                nn.Linear(rnncell_dim*2, n_classes)
            )

    def forward(self, x):
        b,_,t,w,h = x.shape 

        x = x.view(b*t,1,w,h)

        h = self.backbone(x)
        h = h.view(b,t,-1)
        h = self.project(h)
        #z = h.size(-1)
        #h = pack_padded_sequence(h, torch.tensor([b,t+t%2]),
        #                        batch_first=True)
        h,_ = self.rnn(h) #h:(b,t,z)
        #h,_ = pad_packed_sequence(h, torch.tensor([b,t])
        #                        ,batch_first=True)
        
        out = self.fc(h) #out:(b,t,5)

        return out

'''
Ref: https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/3DNet/models/resnet.py

https://reurl.cc/R6dl1Z

'''
class Hemo3DCNN(nn.Module):
    def __init__(self, in_channels=10, n_classes=5):
        pass

    def forward(self, x):
        pass


