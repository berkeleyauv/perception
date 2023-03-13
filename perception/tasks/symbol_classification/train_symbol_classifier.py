import argparse
from symbol_dataset import SymbolDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import math
from model import use_model
import torch.nn as nn
import torch

def compute_loss(predicted, target):
    return nn.functional.cross_entropy(predicted, target)

def compute_accuracy(predicted, target):
    target = torch.argmax(target, dim=1)
    predicted = torch.argmax(predicted, dim=1)
    return torch.sum(predicted == target)

def train(args):
    train_symbol_dataset = SymbolDataset(args.data_folder, duplication_factor=int(math.ceil(args.batch_size / 14.0)))
    eval_symbol_dataset = SymbolDataset(args.data_folder, duplication_factor=3, eval=True)
    optimizer = optim.Adam(use_model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        train_symbol_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    eval_loader = DataLoader(
        eval_symbol_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    for _ in range(args.e):
        train_epoch_loss = torch.tensor(0.0, device=args.device)
        train_accuracy_count = torch.tensor(0, device=args.device)
        eval_epoch_loss = torch.tensor(0.0, device=args.device)
        eval_accuracy_count = torch.tensor(0, device=args.device)
        use_model.train()
        for _, (data, target) in enumerate(train_loader):
            data = data.to(args.device)
            target = target.to(args.device)
            optimizer.zero_grad()
            predictions = use_model(data)
            loss = compute_loss(predictions, target)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.detach()
            train_accuracy_count += compute_accuracy(predictions, target)
        with torch.no_grad():
            use_model.eval()
            for _, (data, target) in enumerate(eval_loader):
                data = data.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)
                predictions = use_model(data)
                loss = compute_loss(predictions, target)
                eval_epoch_loss += loss.detach()
                eval_accuracy_count += compute_accuracy(predictions, target)

        train_epoch_loss = train_epoch_loss.cpu().numpy()
        eval_epoch_loss = eval_epoch_loss.cpu().numpy()
        train_accuracy_count = train_accuracy_count.cpu().numpy()
        eval_accuracy_count = eval_accuracy_count.cpu().numpy()
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_accuracy_count / float(len(train_symbol_dataset)):.3f}")
        print(f"Validation loss: {eval_epoch_loss:.3f}, validation acc: {eval_accuracy_count / float(len(eval_symbol_dataset)):.3f}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_folder", type=str, default="data", help="The absolute path to the data folder for the screenshots of symbols")
    parser.add_argument("-e", type=int, help="The number of epochs to train for the model", default=50)
    parser.add_argument("--num_workers", type=int, help="The number of worksfor getting the batches", default=6)
    parser.add_argument("--batch_size", type=int, help="The batch size for training", default=64)
    parser.add_argument("--device", type=str, help="The device to run the code on, cpu or cuda:number", default="cpu")
    parser.add_argument("--lr", type=float, help="The learning rate", default=1e-3)
    args = parser.parse_args()
    train(args)