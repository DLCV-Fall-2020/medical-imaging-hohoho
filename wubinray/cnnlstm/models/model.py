import torchvision.models as models 
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''
Ref: https://github.com/darraghdog/rsna/blob/master/scripts/trainlstm.py

https://reurl.cc/Ag1egj
'''
resnets = {"resnet18": models.resnet18(pretrained=True),
           "resnet34": models.resnet34(pretrained=True),
           "resnet50": models.resnet50(pretrained=True)}
class HemoCnnLstm(nn.Module):
    def __init__(self, backbone="resnet18", in_channels=10, n_classes=5):
        super().__init__()

        self.resnet = resnets[backbone]
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                    stride=2, padding=3, bias=False)

        self.backbone = nn.Sequential(*list(self.resnet.children())[:-1])
        self.backbone = nn.DataParallel(self.backbone)

        self.lstm = nn.LSTM(self.resnet.fc.in_features, 64, 
                            bidirectional=True, batch_first=True)
        #self.lstm = nn.DataParallel(self.lstm)

        self.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.ReLU(True),
                nn.Linear(64*2, n_classes)
            )

    def forward(self, x):
        b,_,t,w,h = x.shape 

        x = x.view(b*t,1,w,h)

        h = self.backbone(x)
        h = h.view(b,t,-1)
        z = h.size(-1)

        #h = pack_padded_sequence(h, torch.tensor([b,t+t%2]),
        #                        batch_first=True)
        h,_ = self.lstm(h) #h:(b,t,z)
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


