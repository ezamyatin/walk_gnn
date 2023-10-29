from collections import defaultdict
import pandas as pnd
import tqdm
import networkx as nx
import torch


class AdamicAdar:

    def predict(self, feat, edge_index, edge_attr):
        n = len(feat)
        mtr = torch.zeros((n, n), device=torch.device(feat.device))
        mtr[edge_index[0], edge_index[1]] = 1
        mtr[edge_index[1], edge_index[0]] = 1
        out_degree = mtr.sum(dim=1).reshape((-1, 1))
        aa = torch.matmul(mtr / (1 + torch.log(1 + out_degree)), mtr.T)
        return aa

    def eval(self):
        pass


class WeightedAdamicAdar(AdamicAdar):

    def predict(self, feat, edge_index, edge_attr):
        n = len(feat)
        mtr = torch.zeros((n, n), device=torch.device(feat.device))
        mtr[edge_index[0], edge_index[1]] = torch.log(edge_attr + 1).sum(dim=1)
        mtr = mtr + mtr.T
        out_degree = (mtr > 0).sum(dim=1).reshape((-1, 1))
        aa = torch.matmul(mtr / (1 + torch.log(1 + out_degree)), mtr.T)
        return aa