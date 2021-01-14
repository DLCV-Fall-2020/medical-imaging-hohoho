import torchvision.models as models
import torch.nn as nn
import torch

class HemoDenseNet121(nn.Module):
    def __init__(self, in_channels = 3, n_classes=5):
        super().__init__()
        densenet = models.densenet121(pretrained=True, progress=True)
        if in_channels != 3:
            densenet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, 
                                     stride=2, padding=3,bias=False)
        densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(in_features=densenet.classifier.in_features, out_features=n_classes)
        )
        self.base_model = densenet
    def forward(self, x):
        return self.base_model(x)

    def featrue(self, x):
        for layer in list(self.base_model.children())[:-1]:
            x = layer(x)
        return x

