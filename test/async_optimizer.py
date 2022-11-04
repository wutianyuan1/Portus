import torch
import gpu_rpma


class AsyncOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
    
    def step(self):
        gpu_rpma.wait_checkpoint_done()
        return self.optimizer.step()
