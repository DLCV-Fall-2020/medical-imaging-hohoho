import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F

class HemoResNet18(nn.Module):
    def __init__(self, in_channels = 3, n_classes=5):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                                     stride=2, padding=3,bias=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
    def forward(self, x):
        return self.base_model(x)

# https://github.com/darraghdog/rsna/blob/3912ce7a1eeb66a423080be1c4ee9afd6256c170/scripts/trainlstm.py
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class HemoLSTMBasic(nn.Module):
    def __init__(self, embed_size=512, LSTM_UNITS=64, DO = 0.3, n_classes=5):
        super(HemoLSTMBasic, self).__init__()
        
        self.embedding_dropout = SpatialDropout(0.0) #DO)
        
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

if __name__ == "__main__":
    model = HemoLSTMBasic(embed_size=512)
    x = torch.Tensor(10,40,512)
    y = model(x)
    print(y.shape)