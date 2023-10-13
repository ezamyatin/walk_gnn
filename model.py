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


class WalkGNN(LightningModule):
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
        n = feat.shape[0]
        mask = torch.ones((n, n), device=torch.device(feat.device)).bool()
        mask[edge_index[0], edge_index[1]] = False
        mask[edge_index[1], edge_index[0]] = False
        mask[0, :] = False
        mask[:, 0] = False
        mask[torch.arange(0, n), torch.arange(0, n)] = False
        pred = self.forward(feat, edge_index, edge_attr).reshape((n, -1))
        return pred, mask

    def recommend(self, feat, edge_index, edge_attr, k):
        n = feat.shape[0]
        pred, mask = self.predict(feat, edge_index, edge_attr)
        pred[~mask.bool()] = -np.inf
        y_score = pred.reshape(-1).cpu().detach().numpy()
        taken = set()
        recs = []
        for i in y_score.argsort()[::-1][:2*k]:
            if len(recs) == k:
                break
            item = min(i // n, i % n), max(i // n, i % n)
            if item in taken:
                continue
            recs.append(item)
            taken.add(item)
        return recs

    def training_step(self, batch, batch_idx):
        loss = 0
        batch_size = batch[0].shape[0]
        for i in range(batch_size):
            ego_f, f, edge_index, label = batch
            pred, mask = self.predict(ego_f[i], edge_index[i], f[i])
            n = pred.shape[0]

            mask[label[i][:, 0], label[i][:, 1]] = False
            mask[label[i][:, 1], label[i][:, 0]] = False
            meta_y = torch.ones(n, device=torch.device(ego_f.device))
            for u, v in label[i]:
                loss += (self.loss_fn(((pred[u, v].reshape((1)) - pred[u, mask[u]])), meta_y[mask[u]]) / len(label[i])).sum() / n
                loss += (self.loss_fn(((pred[v, u].reshape((1)) - pred[v, mask[v]])), meta_y[mask[v]]) / len(label[i])).sum() / n

        self.log("loss/train", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

