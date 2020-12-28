from torchvision import transforms

def get_default_transform(mode ="test", RGB = False):
    if mode == "train":
        if RGB:
            default_transform = transforms.Compose([
                                    transforms.RandomRotation(20),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                                ])
        else:
            default_transform = transforms.Compose([
                                    #transforms.RandomRotation(20),
                                    #transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                ])

    else:
        if RGB:
            default_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                                ])
        else:
            default_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                ])
    return default_transform
