from torch.utils.data.dataset import IterableDataset, Dataset
import numpy as np
import torch
import os

DATA_PREFIX = './data/'
LIMIT = None


def get_mask(feat, edge_index, edge_attr):
    n = feat.shape[0]
    mask = torch.ones((n, n), device=torch.device(feat.device)).bool()
    mask[edge_index[0], edge_index[1]] = False
    mask[edge_index[1], edge_index[0]] = False
    mask[0, :] = False
    mask[:, 0] = False
    mask[torch.arange(0, n), torch.arange(0, n)] = False
    return mask


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


class InMemoryEgoLabelDataset(Dataset):
    def __init__(self, ego_net_path, label_path, limit=None):
        iter = EgoLabelDataset(ego_net_path, label_path, limit)
        self.data = []
        for ego_id, ego_f, f, edge_index, label in iter:
            self.data.append((ego_id, ego_f, f, edge_index, label))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class YeastDataset(Dataset):
    def __init__(self, yeast_path, train):
        graph_id2label = []
        node_id2label = []
        node_id2graph_id = []
        edge_path = os.path.join(yeast_path, 'Yeast_A.txt')
        node_label_path = os.path.join(yeast_path, 'Yeast_node_labels.txt')
        edge_label_path = os.path.join(yeast_path, 'Yeast_edge_labels.txt')
        graph_indicator_path = os.path.join(yeast_path, 'Yeast_graph_indicator.txt')
        graph_label_path = os.path.join(yeast_path, 'Yeast_graph_labels.txt')

        with open(graph_label_path, 'r') as file:
            for line in file:
                graph_id2label.append(int(line))

        with open(node_label_path, 'r') as file:
            for line in file:
                node_id2label.append(int(line))

        with open(graph_indicator_path, 'r') as file:
            for line in file:
                node_id2graph_id.append(int(line) - 1)

        graph_id2n = [0] * len(graph_id2label)
        node_id2i = [-1] * len(node_id2label)
        data = [(i, [], [], []) for i in range(len(graph_id2label))]

        with open(edge_path, 'r') as file1:
            with open(edge_label_path, 'r') as file2:
                for line1, line2 in zip(file1, file2):
                    a, b = list(map(int, line1.split(", ")))
                    a -= 1
                    b -= 1
                    f = int(line2)

                    assert node_id2graph_id[a] == node_id2graph_id[b]
                    graph_id = node_id2graph_id[a]

                    if train and graph_id % 100 >= 80:
                        continue

                    if not train and graph_id % 100 < 80:
                        continue

                    if node_id2i[a] == -1:
                        node_id2i[a] = graph_id2n[graph_id]
                        data[graph_id][1].append(node_id2label[a])
                        graph_id2n[graph_id] += 1

                    if node_id2i[b] == -1:
                        node_id2i[b] = graph_id2n[graph_id]
                        data[graph_id][1].append(node_id2label[b])
                        graph_id2n[graph_id] += 1
                    data[graph_id][2].append(f)
                    data[graph_id][3].append([node_id2i[a], node_id2i[b]])

        self.data = [x for x in data if len(x[1]) > 0 and len(x[2]) > 0 and len(x[3]) > 0]

    def __getitem__(self, index):
        # zero node is empty
        graph_id, f, edge_f, edge_index = self.data[index]
        f_oh = np.zeros((len(f) + 1, 74), dtype=np.float32)
        f_oh[np.arange(len(f)) + 1, f] = 1

        edge_f_oh = np.zeros((len(edge_f), 3), dtype=np.float32)
        edge_f_oh[np.arange(len(edge_f)), edge_f] = 1

        edge_index = np.array(edge_index, dtype=np.int64) + 1

        label_i = np.random.randint(0, len(edge_index))
        label = np.array([np.min(edge_index[label_i]), np.max(edge_index[label_i])])
        idx = ((np.array(edge_index) != label).sum(axis=1) != 0) & ((np.array(edge_index) != label[::-1]).sum(axis=1) != 0)

        return graph_id, f_oh, edge_f_oh[idx], edge_index[idx].T, [label]

    def __len__(self):
        return len(self.data)