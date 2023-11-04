import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import numpy as np


class MLP(nn.Module):
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
    def __init__(self, edge_dim, hid_dim, mlp_layers):
        super().__init__()
        if edge_dim is not None:
            self.ignore_edge_attr = False
            self.edge_mlp = MLP(edge_dim, hid_dim * 4, hid_dim * hid_dim, mlp_layers)
        else:
            self.ignore_edge_attr = True
            self.edge_mlp = nn.Linear(hid_dim, hid_dim, bias=False)
        self.mlp = MLP(hid_dim, hid_dim * 4, hid_dim, mlp_layers)
        self.linear = nn.Linear(hid_dim, hid_dim)

    def forward(self, mtr, edge_index, edge_attr):
        n = mtr.shape[0]
        hid_dim = mtr.shape[2]

        fmtr = torch.zeros((n, n, hid_dim, hid_dim), device=torch.device(mtr.device))
        if not self.ignore_edge_attr:
            fmtr[edge_index[0], edge_index[1]] = self.edge_mlp(edge_attr).reshape((-1, hid_dim, hid_dim))
            mtr1 = torch.einsum('ijc,jkct->ikt', mtr, fmtr)
        else:
            fmtr[edge_index[0], edge_index[1]] = self.edge_mlp.weight
            mtr1 = torch.einsum('ijc,jkct->ikt', mtr, fmtr)

        mtr1[torch.arange(0, n), torch.arange(0, n), :] *= 0
        mtr1 /= hid_dim
        return mtr + self.mlp(mtr1 + self.linear(mtr1.permute((1, 0, 2))))


class WalkGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks, mlp_layers):
        super().__init__()
        self.hid_dim = hid_dim
        self.blocks = nn.ModuleList([WalkConv(edge_dim, hid_dim, mlp_layers) for _ in range(num_blocks)])
        if node_dim is not None:
            self.ignore_node_attr = False
            self.node_mlp = MLP(node_dim, hid_dim * 2, hid_dim, 2)
        else:
            self.ignore_node_attr = True
            self.node_mlp = torch.nn.Parameter(torch.ones(hid_dim, dtype=torch.float32))
        self.out_mlp = MLP(hid_dim, hid_dim * 2, 1, 2)

    def forward(self, feat, edge_index, edge_attr):
        n = feat.shape[0]
        mtr = torch.zeros((n, n, self.hid_dim), device=torch.device(feat.device))
        if not self.ignore_node_attr:
            mtr[torch.arange(0, n), torch.arange(0, n)] = self.node_mlp(feat)
        else:
            mtr[torch.arange(0, n), torch.arange(0, n)] = self.node_mlp

        for block in self.blocks:
            mtr = block(mtr, edge_index, edge_attr)
        return self.out_mlp(mtr).reshape((n, n))

    def predict(self, feat, edge_index, edge_attr):
        return self.forward(feat, edge_index, edge_attr)

