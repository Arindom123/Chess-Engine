import chess
import torch
import os
from model import ChessNet
from engine import findBestMove
from train import trainEngine
numGames = 100
listBoardStates = []
model = ChessNet()
if os.path.exists("internalWeights.pth"):
    model.load_state_dict(torch.load("internalWeights.pth", weights_only = True))
optimizer = torch.optim.Adam(model.parameters(),lr=.001)
for _ in range(numGames):
    board = chess.Board()
    while not board.is_game_over():
        listBoardStates.append(board.copy())
        board.push(findBestMove(board, model))
    listBoardStates.append(board.copy())
    trainEngine(listBoardStates, model, optimizer)
    listBoardStates = []
torch.save(model.state_dict(), "internalWeights.pth")
