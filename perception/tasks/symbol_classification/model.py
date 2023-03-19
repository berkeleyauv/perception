import torch.nn as nn
import torchvision.models as models
import torch.nn as nn

NUM_CLASSES = 2

def get_model(pretrained, fine_tune, num_classes):
    model = models.efficientnet_v2_s(weights='DEFAULT')
    if fine_tune:
        for params in model.parameters():
            params.requires_grad = True
    else:
        for params in model.parameters():
            params.requires_grad = False
    model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model

use_model = get_model(pretrained=True, fine_tune=True, num_classes=NUM_CLASSES)