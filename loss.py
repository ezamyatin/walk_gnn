import torch
from torch import nn

from dataset import get_mask


class PWLoss:
    def __init__(self):
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def compute_loss(self, model, batch, batch_idx):
        loss = 0
        batch_size = batch[0].shape[0]
        for i in range(batch_size):
            ego_id, feat, edge_attr, edge_index, label = batch
            mask = get_mask(feat[i], edge_index[i], edge_attr[i])
            pred = model.predict(feat[i], edge_index[i], edge_attr[i])
            n = pred.shape[0]
            mask[label[i][:, 0], label[i][:, 1]] = False
            mask[label[i][:, 1], label[i][:, 0]] = False
            meta_y = torch.ones(n, device=torch.device(feat.device))
            for u, v in label[i]:
                loss += (self.loss_fn(((pred[u, v].reshape((1)) - pred[u, mask[u]])), meta_y[mask[u]]) / len(label[i])).sum() / n
                loss += (self.loss_fn(((pred[v, u].reshape((1)) - pred[v, mask[v]])), meta_y[mask[v]]) / len(label[i])).sum() / n
        return loss