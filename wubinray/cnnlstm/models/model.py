from models.densenet import HemoDenseNet121

import torchvision.models as models
import torch.nn.functional as F
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

class SpatialDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.drop2d = nn.Dropout2d(p, inplace=True)
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1, 3)   # convert to [batch, channels, time]
        x = self.drop2d(x)
        x = x.permute(0, 2, 1, 3)   # back to [batch, time, channels]
        x = x.squeeze(-1)
        return x 
    
class HemoLSTMBasic(nn.Module):
    def __init__(self, embed_size=512, LSTM_UNITS=64, DO = 0.3, n_classes=5):
        super(HemoLSTMBasic, self).__init__()
        
        #self.embedding_dropout = SpatialDropout(0.0) #DO)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear = nn.Linear(LSTM_UNITS*2, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x

        #h_embadd = torch.cat((h_embedding[:,:,:2048], h_embedding[:,:,:2048]), -1)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 #+ h_embadd

        output = self.linear(hidden)
        
        return output
    
class HemoCnnLstm(nn.Module):
    def __init__(self, backbone="resnet18", in_channels=10, n_classes=5, 
                    pretrained=None):
        super().__init__()
       
        if "resnet" in backbone:  
            resnet = resnets[backbone](pretrained=True)
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                        stride=2, padding=3, bias=False)
            self.cnn_net = resnet 
            in_features = resnet.fc.in_features 
        elif "densnet121" == backbone:
            densnet = HemoDenseNet121(1, n_classes)
            densnet.load_state_dict(
                            torch.load(pretrained,map_location="cpu"))
            self.cnn_net = densnet.base_model 
            
            in_features = densnet.base_model.classifier.in_features 
        else:
            raise NotImplementedError(f"{backbone} is not implemented")
        # hyper parameters
        EMBED_SIZE=in_features 
        LSTM_UNITS=64
 
        # backbone
        self.backbone = nn.Sequential(*list(self.cnn_net.children())[:-1])
        self.backbone = nn.DataParallel(self.backbone)

        self.drop2d = SpatialDropout(p=0.2)
       
        # lstm
        self.lstm = HemoLSTMBasic(EMBED_SIZE, LSTM_UNITS, DO=0, 
                                                    n_classes=n_classes)
        #self.lstm = nn.DataParallel(self.lstm)

    def forward(self, x):
        b,_,t,w,h = x.shape 
        
        # cnn
        x = x.view(b*t,1,w,h)
        h = self.backbone(x)
        h = h.view(b,t,-1)
        
        if self.training:
            h = self.drop2d(h)
        
        # lstm
        out = self.lstm(h)

        return out # out(b,t_max,5)

'''
Ref: https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/3DNet/models/resnet.py

https://reurl.cc/R6dl1Z

'''
class Hemo3DCNN(nn.Module):
    def __init__(self, in_channels=10, n_classes=5):
        pass

    def forward(self, x):
        pass


