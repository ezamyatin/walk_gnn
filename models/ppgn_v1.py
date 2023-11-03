# code taken from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch
import torch
import torch.nn as nn


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


class PPGN_V1(nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks, mlp_layers):
        super().__init__()
        self.edge_dim = edge_dim
        last_layer_features = edge_dim
        self.reg_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            mlp_block = RegularBlock(mlp_layers, last_layer_features, hid_dim)
            self.reg_blocks.append(mlp_block)
            last_layer_features = hid_dim

        out_layers = []
        out_layers.append(FullyConnected(last_layer_features, hid_dim))
        for i in range(mlp_layers):
            out_layers.append(FullyConnected(hid_dim, hid_dim))
        out_layers.append(FullyConnected(hid_dim, 1, activation_fn=None))
        self.out_mlp = nn.Sequential(*out_layers)

    def forward(self, feat, edge_index, edge_attr):
        x = torch.zeros((feat.shape[0], feat.shape[0], self.edge_dim), device=feat.device, dtype=torch.float32)
        x[edge_index[0], edge_index[1]] = edge_attr
        x = x.permute((2, 0, 1)).unsqueeze(0)
        for i, block in enumerate(self.reg_blocks):
            x = block(x)
        return self.out_mlp(x[0].permute((1, 2, 0))).reshape((feat.shape[0], feat.shape[0]))

    def predict(self, feat, edge_index, edge_attr):
        return self.forward(feat, edge_index, edge_attr)

