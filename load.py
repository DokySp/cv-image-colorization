import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()


model = Test()
model.load_state_dict(torch.load("Best_model_weight.pth"))

print(model)
