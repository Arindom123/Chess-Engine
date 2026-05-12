import chess
import torch
import os
from model import ChessNet
from engine import findBestMove
from train import trainEngine
numSets = 20
autoSave = 50
#total games = numSets*autoSave
listBoardStates = []
model = ChessNet()
if os.path.exists("internalWeights.pth"):
    model.load_state_dict(torch.load("internalWeights.pth", weights_only = True))
optimizer = torch.optim.Adam(model.parameters(),lr=.001)
board = chess.Board()
# while not board.is_game_over():
#     listBoardStates.append(board.copy())
#     board.push_uci(input("Enter your move"))
#     if not board.is_game_over():
#         engineMove = findBestMove(board, model)
#         print(engineMove)
#         board.push(engineMove)
# listBoardStates.append(board.copy())
# trainEngine(listBoardStates, model, optimizer)
# listBoardStates = []
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
    torch.save(model.state_dict(), "internalWeights.pth")
    print (f"Save {setIndex+1} of {numSets}")