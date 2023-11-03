# code taken from https://github.com/GraphPKU/I2GNN/blob/master/count_models.py#L460

import torch.nn as nn
from torch_geometric.utils import dropout_adj, to_dense_adj, to_dense_batch, degree
import torch
from torch_geometric.nn.models import MLP

from models.gin import SimpleNormLayer


class RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    """
    def __init__(self, depth_of_mlp, in_features, out_features):
        super().__init__()

        self.out_features = out_features
        self.mlp1 = MlpBlock(in_features, out_features, depth_of_mlp + 1)
        self.mlp2 = MlpBlock(in_features, out_features, depth_of_mlp + 1)

        self.skip = SkipConnection(in_features+out_features, out_features)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)

        mult = torch.matmul(mlp1, mlp2)
        mult /= self.out_features
        out = self.skip(in1=inputs, in2=mult)
        return out


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        self.out_features = out_features
        for i in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))
            out /= self.out_features

        return out


class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)
        _init_weights(self.conv)

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        out /= self.out_features
        return out


class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=nn.functional.relu):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)

        self.activation = activation_fn

    def forward(self, input):
        out = self.fc(input)
        if self.activation is not None:
            out = self.activation(out)

        return out


def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    # nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)



class PPGN(torch.nn.Module):
    # Provably powerful graph networks
    def __init__(self, emb_dim=64, use_spd=False, y_ndim=1,
                 **kwargs):
        super(PPGN, self).__init__()

        self.use_spd = use_spd
        self.y_ndim = y_ndim

        initial_dim = 2

        if self.use_spd:
            initial_dim += 1

        # ppgn modules
        num_blocks = 2
        num_rb_layers = 4

        self.ppgn_rb = torch.nn.ModuleList()
        self.ppgn_rb.append(RegularBlock(num_blocks, initial_dim, emb_dim))
        for i in range(num_rb_layers - 1):
            self.ppgn_rb.append(RegularBlock(num_blocks, emb_dim, emb_dim))

        self.out_mlp = MLP(
            [emb_dim, emb_dim, 1],
            act='relu',
            act_first=False,
            norm=SimpleNormLayer(emb_dim),
            norm_kwargs=None
        )

    def forward(self, x, edge_index):
        # prepare dense data
        device = edge_index.device
        node_embedding = torch.zeros(x.shape[0]).to(device)
        dense_edge_data = to_dense_adj(edge_index, max_num_nodes=x.shape[0])
        dense_edge_data = torch.unsqueeze(dense_edge_data, -1)
        dense_node_data, mask = to_dense_batch(node_embedding)
        dense_node_data = torch.unsqueeze(dense_node_data, -1)
        shape = dense_node_data.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_data = torch.empty(*shape).to(device)

        if self.use_spd:
            dense_dist_mat = torch.zeros(shape[0], shape[1], shape[1], 1).to(device)

        for g in range(shape[0]):
            for i in range(shape[-1]):
                diag_node_data[g, :, :, i] = torch.diag(dense_node_data[g, :, i])

        if self.use_spd:
            z = torch.cat([dense_dist_mat, dense_edge_data, diag_node_data], -1)
        else:
            z = torch.cat([dense_edge_data, diag_node_data], -1)
        z = torch.transpose(z, 1, 3)

        # ppng
        for rb in self.ppgn_rb:
            z = rb(z)
        return self.out_mlp(z[0].permute((1, 2, 0))).reshape((x.shape[0], x.shape[0]))


class PPGNModel(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.model = PPGN(hid_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.model.forward(x, edge_index)

    def predict(self, x, edge_index, edge_attr):
        return self.forward(x, edge_index, edge_attr)

