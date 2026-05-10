import torch
import chess
from model import ChessNet

def generateTensor(board):
    tensor = torch.zeros(768);

    indexPieces = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
    }
    for square, piece in board.piece_map().items():
        index = indexPieces[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        finalIndex = (color_offset + index)*64 + square
        tensor[finalIndex] = 1.0
    return tensor

def runTest():
    board = chess.Board()
    model = ChessNet()
    input = generateTensor(board)
    inputToList = input.unsqueeze(0)

    with torch.no_grad():
        output = model(input)
    print("eval:", output.item())
if __name__ == "__main__":
    runTest()