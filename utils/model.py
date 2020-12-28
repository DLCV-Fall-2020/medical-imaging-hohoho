import torchvision.models as models
import torch.nn as nn
import torch

class HemoResNet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
    def forward(self, x):
        return self.base_model(x)

class HemoResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
    def forward(self, x):
        return self.base_model(x)

class HemoResnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
    def forward(self, x):
        return self.base_model(x)

if __name__ == "__main__":
    x = torch.Tensor(10,3,512,512)
    model = HemoResNet50(7)
    out = model(x)
    print(out.shape)