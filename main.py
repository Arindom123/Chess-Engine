import chess
from model import ChessNet
from engine import findBestMove

board = chess.Board()
model = ChessNet();
while not board.is_game_over():
    board.push(findBestMove(board, model))
print(board)
print(board.result())