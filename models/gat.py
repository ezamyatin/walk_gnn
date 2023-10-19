from pytorch_lightning import LightningModule
from torch_geometric.nn import GINEConv, GAT
from torch_geometric.nn.models import MLP
import torch
import numpy as np
from torch_geometric.nn.models.basic_gnn import BasicGNN


class GATModel(GAT):

    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks):
        super().__init__(in_channels=hid_dim, hidden_channels=hid_dim, num_layers=num_blocks, edge_dim=hid_dim, v2=True)

        self.in_node_mlp = MLP(
            [300+node_dim, hid_dim * 2, hid_dim],
            act='relu',
            act_first=False,
            norm=None,
            norm_kwargs=None,
        )

        self.in_edge_mlp = MLP(
            [edge_dim, hid_dim * 2, hid_dim],
            act='relu',
            act_first=False,
            norm=None,
            norm_kwargs=None,
        )

    def forward(self, x, edge_index, *args, **kwargs):
        n = x.shape[0]
        one_hot = torch.zeros((300, 300), device=x.device)
        one_hot[torch.arange(n), torch.arange(n)] = 1

        x = self.in_node_mlp(torch.cat([x, one_hot[:n]], dim=1))
        kwargs['edge_attr'] = self.in_edge_mlp(kwargs['edge_attr'])
        return super().forward(x, edge_index, *args, **kwargs)

    def predict(self, feat, edge_index, edge_attr):
        n = feat.shape[0]
        fw = self.forward(feat, edge_index, edge_attr=edge_attr).reshape((n, -1))
        pred = torch.matmul(fw, fw.T)
        return pred
