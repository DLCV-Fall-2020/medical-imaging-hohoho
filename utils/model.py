import torchvision.models as models
import torch.nn as nn
import torch

class HemoResNet18(nn.Module):
    def __init__(self, in_channels = 3, n_classes=5):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
    def forward(self, x):
        return self.base_model(x)

class HemoResNet50(nn.Module):
    def __init__(self, in_channels = 3, n_classes=5):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
    def forward(self, x):
        return self.base_model(x)

class HemoResnext50(nn.Module):
    def __init__(self, in_channels = 3, n_classes=5):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
    def forward(self, x):
        return self.base_model(x)

if __name__ == "__main__":
    x = torch.Tensor(10,3,512,512)
    model = HemoResNet50(in_channels = 3, n_classes=7)
    out = model(x)
    print(out.shape)