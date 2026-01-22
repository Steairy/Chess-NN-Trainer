import chess.pgn
import chess
from multiprocessing import Process, Lock, Value
from stockfish import Stockfish
from basicDB import DB
from multiprocessing import Queue
import pathlib
import os

depth = 6

posLimit = 100000
currentFile = 0
worker_limit = 16

analyzed = Value("i", 0)
analyzed_lock = Lock()
seenglobal = {}

fishPath = f"{pathlib.Path(__file__).parent.resolve()}/stockfish/stockfish"
rawPath = f"{pathlib.Path(__file__).parent.resolve()}/raw/"
savePath = f"{pathlib.Path(__file__).parent.resolve()}/Data/"

db = DB(f"{savePath}file{currentFile}")

game_queue = Queue(maxsize=1000)
write_queue = Queue(maxsize=1000)

def worker():
    seen = {}
    stockfish = Stockfish(fishPath, depth=depth)
    while True:
        game = game_queue.get()
        board = chess.Board()
        for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                if fen not in seen:
                    stockfish.set_fen_position(fen)
                    evaluation = stockfish.get_evaluation()
                    if evaluation["type"] == "cp":
                        write_queue.put((fen, evaluation["value"]))
                    seen[fen] = True

        with analyzed_lock:
            global analyzed
            analyzed.value += 1
    
def analyzeFile(file):
    with open(rawPath+file, "r") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game == None:
                break
            
            game_queue.put(game)
            
            

for file in os.listdir(rawPath):
    Process(target=analyzeFile, args=(file,)).start()


for i in range(worker_limit):
    Process(target=worker).start()


while True:

    fen, value = write_queue.get()
    if fen not in seenglobal:
        db.write(fen, value)
        seenglobal[fen] = True
    
    if len(db.db) > posLimit:
        db.flush()
        currentFile += 1
        db = DB(f"{savePath}file{currentFile}")

    print(f"Analyzed: {analyzed.value}")
