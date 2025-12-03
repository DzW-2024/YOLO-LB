import torch
import torch.nn as nn

class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        # Centralized feature map
        mu = x.mean(dim=[2, 3], keepdim=True).expand_as(x)
        x_centered = x - mu

        x_minus_mu_square = x_centered.pow(2)

        # Normalize and calculate the attention weights
        norm_factor = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        y = x_minus_mu_square / (4 * norm_factor) + 0.5
        attention_map = self.activaton(y)

        return x * attention_map

