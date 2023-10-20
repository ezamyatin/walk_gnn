from pytorch_lightning import LightningModule
from torch_geometric.nn import GINEConv
from torch_geometric.nn.models import MLP
import torch
import numpy as np
from torch_geometric.nn.models.basic_gnn import BasicGNN


class GINEModel(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels, out_channels, **kwargs):
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act='relu',
            act_first=False,
            norm=None,
            norm_kwargs=None,
        )
        return GINEConv(mlp, **kwargs)

    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks):
        super().__init__(in_channels=hid_dim, hidden_channels=hid_dim, num_layers=num_blocks, edge_dim=hid_dim)

        self.in_node_mlp = MLP(
            [node_dim, hid_dim * 2, hid_dim],
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

    def reset_parameters(self):
        super().reset_parameters()
        self.in_node_mlp.reset_parameters()
        self.in_edge_mlp.reset_parameters()

    def forward(self, x, edge_index, *args, **kwargs):
        x = self.in_node_mlp(x)
        kwargs['edge_attr'] = self.in_edge_mlp(kwargs['edge_attr'])
        return super().forward(x, edge_index, *args, **kwargs)

    def predict(self, feat, edge_index, edge_attr):
        n = feat.shape[0]
        fw = self.forward(feat, edge_index, edge_attr=edge_attr).reshape((n, -1))
        pred = torch.matmul(fw, fw.T) / fw.shape[1]
        return pred
