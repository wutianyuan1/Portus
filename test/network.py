import torch.nn as nn
import torch.nn.functional as F

# 这个没啥用 瞎b写的 这不是个CNN
class SBNetwork(nn.Module):
    def __init__(self, m, n) -> None:
        # input: m*n matrix, output: 1*10 tensor
        super().__init__()
        self.m = m
        self.n = n
        self.conv = nn.Conv2d(1, 4, 5, padding=2)
        self.fc1 = nn.Linear(4*m*n, 120)
        self.fc2 = nn.Linear(120, 10)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(-1, 4*self.m*self.n)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
