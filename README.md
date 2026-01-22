# Chess-NN-Trainer
Tool that can be used for training a chess evaluation neural network in pytorch

# What Does It Do
It feeds the game data you gave to the program to stockfish, and then uses the stockfish evaluations to train an evaluation neural net.

# Usage

Important Note: Replace the stockfish folder in the project with a release of stockfish. The file in the repo is a placeholder and the program will not work.

First of all, download the required packages from requirements.txt

You need to place chess games in the raw folder, the games have to be in the .pgn format. You can find chess games in the lichess open database
After placing the chess games, run dataset.py
This will take a lot of time if the stockfish depth is high or there are a lot of games
Note: In dataset.py, you can change the stockfish depth the positions will be evaluated in and how many processes will run.

After running dataset.py, you should see some files appearing in the Data folder. After dataset.py is done with all files or you have already obtained enough positions, close dataset.py and start formatter.py
The formatter will turn all files into pytorch shards. After you see that all of the files have turned into shards(both the countdown stops and the file count is the same as the shard count) you can finally start training the model.
Note: You can change the amount of processes for formatter.py as well.

To train the model, adjust the hyperparameters in the code and the network stack to your liking. The model is named LinearTiny and there you can adjust the linear stack. You can also change the save path.
After adjusting what you want, just run model.py and it will start training the model. It is important to know that the model's state dictionary will be saved, so you will load the state dict when using it.
