import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import numpy as np


class FC(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, n_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_channels, hid_channels))
        layers.append(nn.ReLU())
        for _ in range(n_layers):
            layers.append(nn.Linear(hid_channels, hid_channels))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hid_channels, out_channels))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class WalkConv(nn.Module):
    def __init__(self, edge_dim, hid_dim):
        super().__init__()
        self.edge_fc = FC(edge_dim, hid_dim * 4, hid_dim * hid_dim, 2)
        self.fc = FC(hid_dim, hid_dim * 4, hid_dim, 2)
        self.linear = nn.Linear(hid_dim, hid_dim)

    def forward(self, mtr, edge_index, edge_attr):
        n = mtr.shape[0]
        hid_dim = mtr.shape[2]

        fmtr = torch.zeros((n, n, hid_dim, hid_dim), device=torch.device(mtr.device))
        fmtr[edge_index[0], edge_index[1]] = self.edge_fc(edge_attr).reshape((-1, hid_dim, hid_dim))
        mtr1 = torch.einsum('ijc,jkct->ikt', mtr, fmtr)
        mtr1[torch.arange(0, n), torch.arange(0, n), :] *= 0
        mtr1 /= hid_dim
        return mtr + self.fc(mtr1 + self.linear(mtr1.permute((1, 0, 2))))


class WalkGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks):
        super().__init__()
        self.hid_dim = hid_dim
        self.blocks = nn.ModuleList([WalkConv(edge_dim, hid_dim) for _ in range(num_blocks)])
        self.node_fc = FC(node_dim, hid_dim * 2, hid_dim, 2)
        self.out_fc = FC(hid_dim, hid_dim * 2, 1, 2)

    def forward(self, feat, edge_index, edge_attr):
        n = feat.shape[0]
        mtr = torch.zeros((n, n, self.hid_dim), device=torch.device(feat.device))
        mtr[torch.arange(0, n), torch.arange(0, n)] = self.node_fc(feat)

        for block in self.blocks:
            mtr = block(mtr, edge_index, edge_attr)
        return self.out_fc(mtr).reshape((n, n))

    def predict(self, feat, edge_index, edge_attr):
        return self.forward(feat, edge_index, edge_attr)

