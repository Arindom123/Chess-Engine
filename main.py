import chess
import torch
from engine import findBestMove, instantiateModel, WEIGHTS_PATH
from train import trainEngine
from huggingface_hub import HfApi

numSets = 200
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
        torch.save(model.state_dict(), "pytorch_model.bin")
        api = HfApi()
        api.upload_file(
            path_or_fileobj="pytorch_model.bin",
            path_in_repo="pytorch_model.bin",
            repo_id="ArindomP/chessbot",
            repo_type="model",
            commit_message=f"Updated weights after {autoSave} self-played games"
        )
        print (f"Save {setIndex+1} of {numSets}")
except KeyboardInterrupt:
    print("\ngame interrupted, saving...")
    torch.save(model.state_dict(), WEIGHTS_PATH)