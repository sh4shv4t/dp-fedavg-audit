import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Small CIFAR-10 CNN, ~67k parameters.

    GroupNorm instead of BatchNorm: BatchNorm mixes statistics across samples,
    which breaks the per-client sensitivity argument DP relies on.
    """

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.n1 = nn.GroupNorm(8, 32)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.n2 = nn.GroupNorm(8, 64)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.n3 = nn.GroupNorm(8, 64)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.n1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.n2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.n3(self.c3(x))), 2)
        return self.fc(x.flatten(1))
