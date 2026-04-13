import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pathlib
import os
import time
from math import sqrt
import random
import numpy as np
from utils import fen_to_tensor

learning_rate = 1e-4
batch_size = 4096

shardPath = f"{pathlib.Path(__file__).parent.resolve()}/Shards/"
savePath = f"{pathlib.Path(__file__).parent.resolve()}/NN.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class shardDataset(Dataset):
    def __init__(self, shard):
        data = torch.load(shardPath+shard)
        self.positions = data["position"]
        self.labels = data["evaluation"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.positions[idx].float(), self.labels[idx].float()

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(772, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self,x):
        return self.linear_stack(x)

def train_shard(shard, model, loss_fn, optim):
    dataloader = DataLoader(shardDataset(shard), batch_size=batch_size, num_workers=4, shuffle=True)
    model.train()
    avg = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X).squeeze(-1)

        predWDL = torch.sigmoid(pred/400)
        targetWDL = torch.sigmoid(y/400)
        loss = loss_fn(predWDL, targetWDL)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        avg += loss.item()
    print(f"Loss: {sqrt(avg/len(dataloader))}")

def train_model(model, path):
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    shards_completed = 0
    while True:
        shards = os.listdir(shardPath)
        random.shuffle(shards)
        for shard in shards:
            train_shard(shard, model, loss_fn, optim)

            shards_completed += 1
            print(shards_completed)
            if shards_completed % 50 == 0:
                torch.save(model.state_dict(), path)

model = NN().to(device)
if pathlib.Path(savePath).exists():
    model.load_state_dict(torch.load(savePath))

def print_stats():
    for i, layer in enumerate(model.linear_stack):
        if(i % 2 == 1):
            continue

        with torch.no_grad():
            w_max = layer.weight.abs().max().item()
            b_max = layer.bias.abs().max().item()
            print(f"Weight Abs Max = {w_max:.4f}, Bias Abs Max = {b_max:.4f}")

def export(model):
    w1 = model.linear_stack[0].weight.T.detach().cpu().numpy()
    b1 = model.linear_stack[0].bias.detach().cpu().numpy()
    w2 = model.linear_stack[2].weight.T.detach().cpu().numpy()
    b2 = model.linear_stack[2].bias.detach().cpu().numpy()
    w3 = model.linear_stack[4].weight.flatten().detach().cpu().numpy()
    b3 = model.linear_stack[4].bias.detach().cpu().numpy()

    w1.tofile("W1.bin")
    b1.tofile("B1.bin")
    w2.tofile("W2.bin")
    b2.tofile("B2.bin")
    w3.tofile("W3.bin")
    b3.tofile("B3.bin")


#train_model(model, savePath)
#print(model(fen_to_tensor("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").to(device).float()))
export(model)