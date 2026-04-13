import torch
from basicDB import DB
import pathlib
import os
from multiprocessing import Queue, Process
from utils import fen_to_tensor

savePath = f"{pathlib.Path(__file__).parent.resolve()}/Shards/"
dataPath = f"{pathlib.Path(__file__).parent.resolve()}/Data/"
worker_count = 16
fileQueue = Queue()

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