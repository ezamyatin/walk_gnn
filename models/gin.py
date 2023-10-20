from pytorch_lightning import LightningModule
from torch_geometric.nn import GINEConv
from torch_geometric.nn.models import MLP
import torch
import numpy as np
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.utils import degree


class SimpleNormLayer(torch.nn.Module):
    def __init__(self, denom):
        super().__init__()
        self.denom = denom

    def forward(self, x):
        return x / self.denom


class GINEModel(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels, out_channels, **kwargs):
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act='relu',
            act_first=False,
            norm=SimpleNormLayer(self.hid_dim),
            norm_kwargs=None,
        )
        return GINEConv(mlp, **kwargs)

    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks, use_degree_ohe=False, max_nodes=None):
        assert not use_degree_ohe or max_nodes is not None

        self.max_nodes = max_nodes
        self.hid_dim = hid_dim
        super().__init__(in_channels=hid_dim, hidden_channels=hid_dim,
                         num_layers=num_blocks, edge_dim=hid_dim,
                         norm=SimpleNormLayer(hid_dim))
        if use_degree_ohe:
            self.in_node_mlp = MLP(
                [node_dim + max_nodes, hid_dim * 2, hid_dim],
                act='relu',
                act_first=False,
                norm=SimpleNormLayer(hid_dim),
                norm_kwargs=None,
            )
        else:
            self.in_node_mlp = MLP(
                [node_dim, hid_dim * 2, hid_dim],
                act='relu',
                act_first=False,
                norm=SimpleNormLayer(hid_dim),
                norm_kwargs=None,
            )

        self.in_edge_mlp = MLP(
            [edge_dim, hid_dim * 2, hid_dim],
            act='relu',
            act_first=False,
            norm=SimpleNormLayer(hid_dim),
            norm_kwargs=None,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.in_node_mlp.reset_parameters()
        self.in_edge_mlp.reset_parameters()

    def forward(self, x, edge_index, *args, **kwargs):
        d = degree(edge_index[0], self.max_nodes, dtype=torch.int32)
        one_hot = torch.zeros((x.shape[0], self.max_nodes), device=x.device)
        one_hot[torch.arange(x.shape[0]), d[:x.shape[0]]] = 1
        x = self.in_node_mlp(torch.cat([x, one_hot], dim=1))
        kwargs['edge_attr'] = self.in_edge_mlp(kwargs['edge_attr'])
        return super().forward(x, edge_index, *args, **kwargs)

    def predict(self, feat, edge_index, edge_attr):
        n = feat.shape[0]
        fw = self.forward(feat, edge_index, edge_attr=edge_attr).reshape((n, -1))
        pred = torch.matmul(fw, fw.T)
        return pred
