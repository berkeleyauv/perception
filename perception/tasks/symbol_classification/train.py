import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.models as models
import torch.nn as nn

def save_model(epochs, model, optimizer, criterion, pretrained):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"../model_pretrained_{pretrained}.pth")