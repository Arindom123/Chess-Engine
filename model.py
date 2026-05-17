from huggingface_hub import PyTorchModelHubMixin
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module, PyTorchModelHubMixin):
    def __init__ (self):
        super(ChessNet, self).__init__()
        self.layer1 = nn.Linear(768,256)
        self.layer2 = nn.Linear(256,1)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return x