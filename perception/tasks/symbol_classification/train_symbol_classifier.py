import argparse
from symbol_dataset import SymbolDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import math
from model import use_model
import torch.nn as nn

def compute_loss(predicted, target):
    return nn.functional.cross_entropy(predicted, target)

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
    # eval_loader = DataLoader(
    #     eval_symbol_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )
    for _ in range(args.e):
        train_epoch_loss = 0.0
        for batch_no, (data, target) in enumerate(train_loader):
            data = data.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            predictions = use_model(data)
            loss = compute_loss(predictions, target)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.detach().cpu().numpy()

        print(f"Training loss: {train_epoch_loss:.3f}")
        # print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    
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