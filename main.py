import chess
import torch
from engine import findBestMove, instantiateModel, WEIGHTS_PATH
from train import trainEngine
numSets = 2
autoSave = 100
#total games = numSets*autoSave
listBoardStates = []
model = instantiateModel()
optimizer = torch.optim.Adam(model.parameters(),lr=.001)
board = chess.Board()
try:
    for setIndex in range(numSets):
        for gameIndex in range(autoSave):
            board = chess.Board()
            while not board.is_game_over():
                listBoardStates.append(board.copy())
                board.push(findBestMove(board, model))
            listBoardStates.append(board.copy())
            trainEngine(listBoardStates, model, optimizer)
            listBoardStates = []
            print (f"Set {setIndex+1} Game {gameIndex+1} : " + board.result())
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print (f"Save {setIndex+1} of {numSets}")
except KeyboardInterrupt:
    print("\ngame interrupted, saving...")
    torch.save(model.state_dict(), WEIGHTS_PATH)