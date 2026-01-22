import torch
from basicDB import DB
import pathlib
import os
from multiprocessing import Queue, Process

savePath = f"{pathlib.Path(__file__).parent.resolve()}/Shards/"
dataPath = f"{pathlib.Path(__file__).parent.resolve()}/Data/"
worker_count = 16
fileQueue = Queue()

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
    return board

def fen_to_tensor_multidim(fen:str):
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
    board = torch.zeros(size=(8,8,17), dtype=torch.uint8)
    currentX = 0
    currentY = 0
    for letter in tokens[0]:
        if letter.isdigit():
            currentX += int(letter)
        
        elif letter == "/":
            currentX = 0
            currentY += 1
        
        else:
            board[currentY][currentX][pieceTable[letter]] = 1
            currentX += 1
    
    board[:,:,12].fill_(1 if tokens[1] == "w" else 0)

    for letter in tokens[2]:
        if letter == "K":
             board[:,:,13].fill_(1)
        if letter == "Q":
            board[:,:,14].fill_(1)
        if letter == "k":
            board[:,:,15].fill_(1)
        if letter == "q":
            board[:,:,16].fill_(1)
    return board



def saveShard(idx):
    position_data = []
    evaluation_data = []
    db = DB(f"{dataPath}file{idx}")
    for key in db.db:
        position_data.append(fen_to_tensor(key))
        evaluation_data.append(float(db.db[key]))
    
    saved = {
        "position":torch.stack(position_data),
        "evaluation":torch.tensor(evaluation_data, dtype=torch.float32)
    }
    torch.save(saved, savePath+f"shard{idx}.pt")

def worker():
    while not fileQueue.empty():
        idx = fileQueue.get()
        saveShard(idx)
        print(fileQueue.qsize())

for file in os.listdir(dataPath):
    fileQueue.put(file[4:])

for i in range(worker_count):
    Process(target=worker).start()
