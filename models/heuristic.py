from collections import defaultdict
import pandas as pnd
import tqdm
import networkx as nx
import numpy as np


class AdamicAdar:
    def recommend(self, feat, edge_index, edge_attr, k):
        n = len(feat)
        mtr = np.zeros((n, n))
        mtr[edge_index[0], edge_index[1]] = 1
        mtr[edge_index[1], edge_index[0]] = 1
        out_degree = mtr.sum(axis=1).reshape((-1, 1))
        aa = (mtr / (1 + np.log(1 + out_degree))).dot(mtr.T)
        aa[edge_index[0], edge_index[1]] = -np.inf
        aa[edge_index[1], edge_index[0]] = -np.inf
        aa[0, :] = -np.inf
        aa[:, 0] = -np.inf
        aa[np.arange(0, n), np.arange(0, n)] = -np.inf

        taken = set()
        recs = []
        for i in aa.flatten().argsort()[::-1]:
            u, v = min(i // n, i % n), max(i // n, i % n)
            if (u, v) not in taken:
                taken.add((u, v))
                recs.append((u, v))
            if len(recs) == k:
                break
        return recs