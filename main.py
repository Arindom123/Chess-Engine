import chess
from model import ChessNet
from engine import findBestMove
from train import trainEngine
listBoardStates = []
model = ChessNet();
board = chess.Board();
while not board.is_game_over():
    board.push(findBestMove(board, model))
    listBoardStates.append[board]
trainEngine(listBoardStates, 1, model)