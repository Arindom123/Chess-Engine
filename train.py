import chess
import torch
from main import model
from engine import findBestMove

optimizer = torch.optim.Adam(model.parameters(),lr=.001)

def trainEngine(listBoardStates, numTimes, model):
    for _ in range (numTimes):
        optimizer.zero_grad()
        groundTruth = listBoardStates[-1].result();
        for board in listBoardStates:
            numEvals = 0
            numEvals = numEvals + 1
            predictedEval = model(board)
            squaredErrorSums += (predictedEval - groundTruth)**2
        meanSquaredError = squaredErrorSums/numEvals
        meanSquaredError.backward()
        optimizer.step()