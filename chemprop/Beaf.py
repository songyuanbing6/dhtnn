import os

import torch
import torch.nn as nn
import torch.nn.functional as F


from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Beaf(nn.Module):
    def __init__(self):
        super().__init__()
        print("Beaf activation loaded...")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))-0.002
        return x


beaf = Beaf()
x = torch.linspace(-10, 10, 1000)
y = beaf(x)
plt.plot(x, y)
plt.grid()
plt.show()
