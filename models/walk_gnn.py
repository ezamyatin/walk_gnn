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
    def __init__(self, node_dim, edge_dim, hid_dim, mlp_layers):
        super().__init__()

        if edge_dim is None and node_dim is None:
            self.ignore_edge_attr = True
            self.ignore_node_attr = True
            edge_input_dim = 1
            out_dim = hid_dim * 2 + 1
        elif edge_dim is None:
            self.ignore_edge_attr = True
            self.ignore_node_attr = False
            edge_input_dim = node_dim * 2
            out_dim = hid_dim * 2 + node_dim * 2 + 1
        elif node_dim is None:
            self.ignore_edge_attr = False
            self.ignore_node_attr = True
            edge_input_dim = edge_dim
            out_dim = hid_dim * 2 + 1
        else:
            self.ignore_edge_attr = False
            self.ignore_node_attr = False
            edge_input_dim = edge_dim + node_dim * 2
            out_dim = hid_dim * 2 + node_dim * 2 + 1

        self.edge_mlp = MLP(edge_input_dim, hid_dim * 4, hid_dim * hid_dim, mlp_layers)
        self.out_mlp = MLP(out_dim, hid_dim * 4, hid_dim, mlp_layers)

    def forward(self, mtr, feat, edge_index, edge_attr):
        n = mtr.shape[0]
        hid_dim = mtr.shape[2]

        fmtr = torch.zeros((n, n, hid_dim, hid_dim), device=torch.device(mtr.device))
        if self.ignore_node_attr and self.ignore_edge_attr:
            fmtr[edge_index[0], edge_index[1]] = self.edge_mlp(torch.ones(1, device=torch.device(mtr.device))).reshape((hid_dim, hid_dim))
        elif self.ignore_edge_attr:
            fmtr[edge_index[0], edge_index[1]] = self.edge_mlp(torch.cat((feat[edge_index[0]], feat[edge_index[1]]), dim=-1)).reshape((-1, hid_dim, hid_dim))
        elif self.ignore_node_attr:
            fmtr[edge_index[0], edge_index[1]] = self.edge_mlp(edge_attr).reshape((-1, hid_dim, hid_dim))
        else:
            fmtr[edge_index[0], edge_index[1]] = self.edge_mlp(torch.cat((edge_attr, feat[edge_index[0]], feat[edge_index[1]]), dim=-1)).reshape((-1, hid_dim, hid_dim))

        mtr1 = torch.einsum('ijc,jkct->ikt', mtr, fmtr)
        mtr1 /= hid_dim
        mtr1 = torch.cat((mtr1,
                          mtr1.permute((1, 0, 2)),
                          feat.reshape((1, n, -1)).repeat(n, 1, 1),
                          feat.reshape((n, 1, -1)).repeat(1, n, 1),
                          torch.diag(torch.ones(n, device=mtr1.device)).reshape((n, n, 1))), dim=-1)

        return mtr + self.out_mlp(mtr1)


class WalkGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks, mlp_layers):
        super().__init__()
        self.hid_dim = hid_dim
        self.blocks = nn.ModuleList([WalkConv(node_dim, edge_dim, hid_dim, mlp_layers) for _ in range(num_blocks)])
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
            mtr = block(mtr, feat, edge_index, edge_attr)
        return self.out_mlp(mtr).reshape((n, n))

    def predict(self, feat, edge_index, edge_attr):
        return self.forward(feat, edge_index, edge_attr)
