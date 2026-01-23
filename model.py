import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pathlib
import os
import time
from math import sqrt
import random
import numpy as np

learning_rate = 1e-4
batch_size = 4096

shardPath = f"{pathlib.Path(__file__).parent.resolve()}/Shards/"
savePath = f"{pathlib.Path(__file__).parent.resolve()}/linearTinyWeights.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class shardDataset(Dataset):
    def __init__(self, shard, multidim=False):
        data = torch.load(shardPath+shard)
        self.positions = data["position"]
        self.labels = data["evaluation"]
        if multidim:
            self.positions = torch.permute(self.positions, (0, 3, 1, 2))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.positions[idx].float(), self.labels[idx].float()/1000 # normalize

class linearTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(773, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
    
    def forward(self,x):
        return self.linear_stack(x)

def fen_to_tensor(fen:str):
        pieceTable = {
            "p": 0,
            "n": 1,
            "b": 2,
            "r": 3,
            "q": 4,
            "k": 5,
            "P": 6,
            "N": 7,
            "B": 8,
            "R": 9,
            "Q": 10,
            "K": 11
        }
        tokens = fen.split(" ")
        board = torch.zeros(size=(773,), dtype=torch.uint8)
        current = 0
        for letter in tokens[0]:
            if letter.isdigit():
                current += int(letter)
            
            elif letter == "/":
                continue
            
            else:
                board[current*12 + pieceTable[letter]] = 1
                current += 1
        
        board[768] = 1 if tokens[1] == "w" else 0
        for letter in tokens[2]:
            if letter == "K":
                board[769] = 1
            if letter == "Q":
                board[770] = 1
            if letter == "k":
                board[771] = 1
            if letter == "q":
                board[772] = 1
        return board.float().to(device)


def train_shard(shard, model, loss_fn, optim):
    dataloader = DataLoader(shardDataset(shard), batch_size=batch_size, num_workers=4, shuffle=True)
    model.train()
    avg = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X).squeeze(-1)
        loss = loss_fn(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        avg += loss.item()
    print(f"Loss: {sqrt(avg/len(dataloader))*1000}") # turn back normalization and mse


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

model = linearTiny().to(device)
if pathlib.Path(savePath).exists():
    model.load_state_dict(torch.load(savePath))
train_model(model, savePath)
