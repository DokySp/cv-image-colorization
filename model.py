import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()

