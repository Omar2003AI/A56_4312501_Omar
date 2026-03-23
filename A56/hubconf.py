import torch
from train import TinyImageNet_CNN

def AlexNet(pretrained=False):
    model = TinyImageNet_CNN()

    if pretrained:
        model.load_state_dict(
            torch.load("best_tinyimagenet_model_15epochs.pth", map_location="cpu")
        )

    return model