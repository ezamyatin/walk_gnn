import pytorch_lightning as pl
import torch
from torch.utils.data.dataset import IterableDataset
from torch import nn
import numpy as np
import pandas as pd
import tqdm

from model import WalkGNN

DATA_PREFIX = '/home/e.zamyatin/walk_gnn/data/'


class EgoDataset(IterableDataset):
    def __init__(self, ego_net_path, limit=None):
        self.ego_net_path = ego_net_path
        self.limit = limit

    def read_ego_net(self, ego_net_path):
        cur_ego_id = -1
        cur_ego_net = None
        node_attr = None
        was = None

        n = 0
        with open(ego_net_path, 'r') as ego_net_f:
            ego_net_f.readline()
            for ego_line in ego_net_f:
                if self.limit is not None and n == self.limit:
                    break
                ego_line = ego_line.split(',')
                ego_id, u, v, t, x1, x2, x3 = int(ego_line[0]), int(ego_line[1]), int(ego_line[2]), int(
                    ego_line[3]), float(ego_line[4]), float(ego_line[5]), float(ego_line[6])

                if t >= 0:
                    t = 28 / (t + 1)
                else:
                    t = 0

                if ego_id != cur_ego_id:
                    if cur_ego_id != -1:
                        assert (node_attr[was.sum():] != 0).sum() == 0
                        yield cur_ego_id, (node_attr[:was.sum()], np.array(cur_ego_net[0]).T, np.array(cur_ego_net[1]))
                        n += 1
                    cur_ego_id = ego_id
                    cur_ego_net = ([], [])
                    node_attr = np.zeros((300, 8), dtype=np.float32)
                    was = np.zeros(300, dtype=np.int32)

                was[[u, v]] = 1
                if u == 0:
                    node_attr[v, :4] = [t, x1, x2, x3]
                elif v == 0:
                    node_attr[u, 4:8] = [t, x1, x2, x3]
                else:
                    cur_ego_net[0].append([u, v])
                    cur_ego_net[1].append([t, x1, x2, x3])

            if len(cur_ego_net[0]) > 0 and cur_ego_id != -1 and (self.limit is None or n < self.limit):
                yield cur_ego_id, (node_attr[:was.sum()], np.array(cur_ego_net[0]).T, np.array(cur_ego_net[1]))

    def __iter__(self):
        ego_iter = self.read_ego_net(self.ego_net_path)
        for ego_id, ego_net in ego_iter:
            ego_f = ego_net[0].astype("float32")
            f = ego_net[2].astype("float32")
            edge_index = ego_net[1].astype("int64")
            yield ego_id, ego_f, f, edge_index


class LabelDataset(IterableDataset):
    def __init__(self, label_path, limit=None):
        self.label_path = label_path
        self.limit = limit

    def read_label(self, label_path):
        n = 0
        with open(label_path, 'r') as label_f:
            label_f.readline()
            cur_ego_id = -1
            cur_label = None
            for line in label_f:
                if self.limit is not None and n == self.limit:
                    break
                ego_id, u, v = list(map(int, line.split(",")))
                if ego_id != cur_ego_id:
                    if cur_ego_id != -1:
                        yield cur_ego_id, np.array(cur_label)
                        n += 1
                    cur_ego_id, cur_label = ego_id, []
                cur_label.append([u, v])
            if cur_ego_id != -1 and len(cur_label) > 0 and self.limit is not None and n < self.limit:
                yield cur_ego_id, np.array(cur_label)

    def __iter__(self):
        label_iter = self.read_label(self.label_path)
        for ego_id, label in label_iter:
            label = label.astype("int64")
            yield ego_id, label


class EgoLabelDataset(IterableDataset):
    def __init__(self, ego_net_path, label_path, limit=None):
        self.ego_net_path = ego_net_path
        self.label_path = label_path
        self.limit = limit

    def __iter__(self):
        ego_iter = EgoDataset(self.ego_net_path, self.limit)
        label_iter = LabelDataset(self.label_path, self.limit)
        for (ego_id1, ego_f, f, edge_index), (ego_id2, label) in zip(ego_iter, label_iter):
            assert ego_id1 == ego_id2
            yield ego_id1, ego_f, f, edge_index, label
