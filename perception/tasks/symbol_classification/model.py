import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.models as models
import torch.nn as nn

NUM_CLASSES = 2

def get_model(pretrained, fine_tune, num_classes):
    model = models.efficientnet_v2_s(pretrained=pretrained)
    if fine_tune:
        for params in model.parameters():
            params.requires_grad = True
    else:
        for params in model.parameters():
            params.requires_grad = False
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model

model = get_model(pretrained=True, fine_tune=True, num_classes=NUM_CLASSES)