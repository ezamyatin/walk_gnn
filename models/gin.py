from pytorch_lightning import LightningModule
from torch_geometric.nn import GINEConv, GINConv
from torch_geometric.nn.models import MLP
import torch
import numpy as np
from torch_geometric.nn.models.basic_gnn import BasicGNN, GIN
from torch_geometric.utils import degree


class SimpleNormLayer(torch.nn.Module):
    def __init__(self, denom):
        super().__init__()
        self.denom = denom

    def forward(self, x):
        return x / self.denom


class rGINConv(GINConv):

    def forward(self, x, edge_index, size = None):
        rnd = torch.randint(0, 100, (x.shape[0], 1), dtype=torch.float32, device=x.device) / 100
        x = torch.cat((x, rnd), dim=1)
        return super().forward(x, edge_index, size)


class rGINEConv(GINEConv):

    def forward(self, x, edge_index, edge_attr, size=None):
        rnd = torch.randint(0, 100, (x.shape[0], 1), dtype=torch.float32, device=x.device) / 100
        x = torch.cat((x, rnd), dim=1)
        return super().forward(x, edge_index, size)


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
        if not self.r_version:
            return GINEConv(mlp, **kwargs)
        else:
            return rGINEConv(mlp, **kwargs)

    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks, use_degree_ohe=False, max_nodes=None, r_version=False):
        assert not use_degree_ohe or max_nodes is not None

        self.r_version = r_version
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
        if self.max_nodes is not None:
            d = degree(edge_index[0], self.max_nodes, dtype=torch.int32)
            one_hot = torch.zeros((x.shape[0], self.max_nodes), device=x.device)
            one_hot[torch.arange(x.shape[0]), d[:x.shape[0]]] = 1
            x = self.in_node_mlp(torch.cat([x, one_hot], dim=1))
        else:
            x = self.in_node_mlp(x)
        kwargs['edge_attr'] = self.in_edge_mlp(kwargs['edge_attr'])
        return super().forward(x, edge_index, *args, **kwargs)

    def predict(self, feat, edge_index, edge_attr):
        n = feat.shape[0]
        fw = self.forward(feat, edge_index, edge_attr=edge_attr).reshape((n, -1))
        pred = torch.matmul(fw, fw.T)
        return pred


class GINModel(GIN):

    def __init__(self, hid_dim, num_blocks, use_degree_ohe=False, max_nodes=None):
        assert not use_degree_ohe or max_nodes is not None

        self.use_degree_ohe = use_degree_ohe
        self.max_nodes = max_nodes
        self.hid_dim = hid_dim
        super().__init__(in_channels=max_nodes if use_degree_ohe else 1, hidden_channels=hid_dim,
                         num_layers=num_blocks, norm=SimpleNormLayer(hid_dim))

    def forward(self, x, edge_index, *args, **kwargs):
        if self.use_degree_ohe:
            d = degree(edge_index[0], self.max_nodes, dtype=torch.int32)
            one_hot = torch.zeros((x.shape[0], self.max_nodes), device=x.device)
            one_hot[torch.arange(x.shape[0]), d[:x.shape[0]]] = 1
            return super().forward(one_hot, edge_index, *args, **kwargs)
        else:
            return super().forward(torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device), edge_index, *args, **kwargs)

    def predict(self, feat, edge_index, edge_attr):
        n = feat.shape[0]
        fw = self.forward(feat, edge_index).reshape((n, -1))
        pred = torch.matmul(fw, fw.T)
        return pred
