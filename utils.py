import torch

def fen_to_tensor(fen:str):
        tokens = fen.split(" ")
        white = tokens[1] == 'w'
        add = 0 if white else 6
        pieceTable = {
            "p": 6-add,
            "n": 7-add,
            "b": 8-add,
            "r": 9-add,
            "q": 10-add,
            "k": 11-add,
            "P": 0+add,
            "N": 1+add,
            "B": 2+add,
            "R": 3+add,
            "Q": 4+add,
            "K": 5+add
        }
        board = torch.zeros(size=(772,), dtype=torch.uint8)
        current = 56
        for letter in tokens[0]:
            if letter.isdigit():
                current += int(letter)
            
            elif letter == "/":
                current -= 16
                continue
            
            else:
                ind = (current)*12 + pieceTable[letter]
                if not white:
                    ind = (current^56)*12 + pieceTable[letter]
                board[ind] = 1
                current += 1
        
        castling = tokens[2]
        if white:
            if "K" in castling: board[768] = 1
            if "Q" in castling: board[769] = 1
            if "k" in castling: board[770] = 1
            if "q" in castling: board[771] = 1
        else:
            if "k" in castling: board[768] = 1
            if "q" in castling: board[769] = 1
            if "K" in castling: board[770] = 1
            if "Q" in castling: board[771] = 1
        return board