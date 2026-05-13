import chess
import torch
from engine import findBestMove 
from engine import instantiateModel
from train import trainEngine
numSets = 20
autoSave = 50
#total games = numSets*autoSave
listBoardStates = []
model = instantiateModel()
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