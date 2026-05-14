from engine import generateTensor
from engine import boardToTensorList

def trainEngine(listBoardStates, model, optimizer):
    optimizer.zero_grad()
    for state in listBoardStates:
        results = state[-1].result()
        groundTruth = 0
        if len(results) < 6:
            groundTruth = -1.0 if int(results[0]) == 0 else 1.0
        else: groundTruth = 0.0
        numEvals = 0
        squaredErrorSums = 0
        for board in state:
            numEvals = numEvals + 1
            predictedEval = model(boardToTensorList(board))
            squaredErrorSums += (predictedEval - groundTruth)**2
        meanSquaredError = squaredErrorSums/numEvals
        meanSquaredError.backward()
        optimizer.step()