import torch
from engine import findBestMove

def trainEngine(listBoardStates, model, optimizer):
    optimizer.zero_grad()
    results = listBoardStates[-1].result()
    groundTruth = 0
    if len(results) < 6:
        groundTruth = -1 if results[1] == 0 else 1
    else: groundTruth = 0
    numEvals = 0
    squaredErrorSums = 0
    for board in listBoardStates:
        numEvals = numEvals + 1
        predictedEval = model(board)
        squaredErrorSums += (predictedEval - groundTruth)**2
    meanSquaredError = squaredErrorSums/numEvals
    meanSquaredError.backward()
    optimizer.step()