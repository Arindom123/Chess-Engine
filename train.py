from engine import boardToTensorList
import torch
import torch.nn.functional as F

def trainEngine(listBoardStates, model, optimizer):
    optimizer.zero_grad()
    results = listBoardStates[-1].result()
    groundTruth = 0
    groundTruths = []
    if len(results) < 6:
        groundTruth = -1.0 if int(results[0]) == 0 else 1.0
    else: groundTruth = 0.0
    predictedEvals = model(torch.stack(boardToTensorList(listBoardStates)))
    groundTruths = torch.full_like(predictedEvals, groundTruth)
    F.mse_loss(predictedEvals, groundTruths).backward()
    optimizer.step()