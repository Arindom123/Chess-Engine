import torch
import chess

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

def findBestMove(board, model):
    currentTurnWhite = board.turn
    possibleMoves = board.legal_moves
    if currentTurnWhite:
        bestEval = float('-inf')
    else:
        bestEval = float('inf')
    bestMove = None
    for move in possibleMoves:
        board.push(move)
        with torch.no_grad():
            output = model(boardToTensorList(board))
        currentEval = output.item()
        if currentTurnWhite and currentEval > bestEval:
            bestEval = currentEval
            bestMove = move
        elif not currentTurnWhite and currentEval < bestEval:
            bestEval = currentEval
            bestMove = move
        board.pop()
    return bestMove

def boardToTensorList(board):
    boardToTensor = generateTensor(board)
    tensorList = boardToTensor.unsqueeze(0)
    return tensorList