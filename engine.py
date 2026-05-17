import torch
import chess
import random
from model import ChessNet
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "internalWeights.pth")

indexPieces = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def generateTensor(board):
    tensor = torch.zeros(768)
    for square, piece in board.piece_map().items():
        index = indexPieces[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        finalIndex = (color_offset + index)*64 + square
        tensor[finalIndex] = 1.0
    return tensor

def findBestMove(board, model):
    bestMove = None
    percentRandom = 10
    currentTurnWhite = board.turn
    possibleMoves = list(board.legal_moves)
    possibleTensorPositions = []
    if currentTurnWhite:
        bestEval = float('-inf')
    else:
        bestEval = float('inf')
    if random.randint(0,99) < percentRandom:
        return random.choice(possibleMoves)
    for move in possibleMoves:
        board.push(move)
        possibleTensorPositions.append(generateTensor(board))
        board.pop()
    stackedPositions = torch.stack(possibleTensorPositions)
    with torch.no_grad():
        output = model(stackedPositions)
    evalsAndMoves = zip(output, possibleMoves)
    for eval, move in evalsAndMoves:
        currentEval = eval.item()
        if currentTurnWhite and currentEval > bestEval:
            bestEval = currentEval
            bestMove = move
        elif not currentTurnWhite and currentEval < bestEval:
            bestEval = currentEval
            bestMove = move
    return bestMove

def boardToTensorList(boardStates):
    tensorList = []
    for board in boardStates:
        tensorList.append(generateTensor(board))
    return tensorList

def instantiateModel():
    model = ChessNet.from_pretrained("ArindomP/chessbot")
    return model