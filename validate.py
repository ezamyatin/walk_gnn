import argparse

import numpy as np
import pandas as pd
import torch
import tqdm

from dataset import EgoDataset, DATA_PREFIX, LIMIT, YeastDataset
from dataset import get_mask
from models import get_model

NDCG_AT_K = 5


def ndcg_at_k(label_df, subm_df, k, private):
    if private:
        label_df = label_df[label_df['is_private']]
    else:
        label_df = label_df[~label_df['is_private']]

    assert len(subm_df.drop_duplicates(['ego_id', 'u', 'v'])) == len(subm_df)
    assert (subm_df.groupby('ego_id').count() > k).sum().sum() == 0
    assert (label_df['u'] >= label_df['v']).sum() == 0
    subm_df['rank'] = (subm_df.groupby('ego_id').cumcount() + 1)
    df = pd.merge(label_df, subm_df, on=['ego_id', 'u', 'v'], how='left', indicator=True)
    df['hit'] = (df['_merge'] == 'both') * 1.0

    df['idcg'] = 1 / np.log2((df.groupby('ego_id').cumcount() + 1) + 1)
    df['dcg'] = 0
    df.loc[~df['rank'].isna(), 'dcg'] = 1 / np.log2(df['rank'] + 1)
    grouped_df = df.groupby('ego_id').sum()
    mean = (grouped_df['dcg'] / grouped_df['idcg']).mean()
    std = (grouped_df['dcg'] / grouped_df['idcg']).std()
    return mean, 1.96 * std / len(grouped_df) ** 0.5


def recommend(model, feat, edge_index, edge_attr, k):
    n = feat.shape[0]
    pred = model.predict(feat, edge_index, edge_attr)
    mask = get_mask(feat, edge_index, edge_attr)
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


def validate(model, test_ego_path, test_label_path, k, private, device):
    label_df = pd.read_csv(test_label_path)
    ego_ids = set(label_df[label_df['is_private'] == private]['ego_id'])

    out_df = {
        'ego_id': [],
        'u': [],
        'v': [],
    }

    for ego_id, feat, edge_attr, edge_index in tqdm.tqdm(EgoDataset(test_ego_path, LIMIT), total=len(ego_ids) * 2):
        if ego_id not in ego_ids: continue
        feat = torch.tensor(feat, device=device)
        edge_attr = torch.tensor(edge_attr, device=device)
        edge_index = torch.tensor(edge_index, device=device)

        recs = recommend(model, feat, edge_index, edge_attr, k)
        assert len(recs) == k
        for u, v in recs:
            out_df['ego_id'].append(ego_id)
            out_df['u'].append(u)
            out_df['v'].append(v)
    return ndcg_at_k(label_df, pd.DataFrame.from_dict(out_df), k, private)


def ndcg_(model, feat, edge_attr, edge_index, label, k):
    recs = recommend(model, feat, edge_index, edge_attr, k)
    assert len(recs) <= k
    dcg = 0
    idcg = 0

    for i, rec in enumerate(recs):
        if ((rec[0] == label[:, 0]) & (rec[1] == label[:, 1])).any():
            dcg += 1/np.log2(i+2)
        if i < len(label):
            idcg += 1/np.log2(i+2)
    return dcg/idcg


def auc_(model, feat, edge_attr, edge_index, label):
    n = len(feat)
    recs = recommend(model, feat, edge_index, edge_attr, n * (n-1) // 2)

    targets = np.zeros(len(recs))

    for i, rec in enumerate(recs):
        if ((rec[0] == label[:, 0]) & (rec[1] == label[:, 1])).any():
            targets[i] = 1
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(targets, np.arange(len(targets))[::-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['ego-vk', 'yeast'])
    parser.add_argument('--dataset_path')
    parser.add_argument('--model', choices=['aa', 'waa', 'walk_gnn', 'walk_gnn_no_attr', 'walk_gnn_no_node_attr', 'walk_gnn_no_edge_attr',
                                            'gine', 'gine_ohe', 'gin_ohe', 'gin_constant', 'ppgn', 'ppgn_no_attr',
                                            'walk_gnn_2b', 'walk_gnn_4b', 'walk_gnn_8b'])
    parser.add_argument('--state_dict_path', default=None)
    parser.add_argument('--device', choices=['cpu'] + ['cuda:{}'.format(i) for i in range(4)])
    args = parser.parse_args()
    if args.task == 'ego-vk':
        model = get_model(args, 8, 4)
    elif args.task == 'yeast':
        model = get_model(args, 74, 3)
    else:
        assert False
    model.eval()

    with torch.no_grad():
        if args.task == 'ego-vk':
            metric, confidence = validate(model, DATA_PREFIX + "ego_net_te.csv", DATA_PREFIX + "val_te_pr.csv", NDCG_AT_K, True, device=torch.device(args.device))
        elif args.task == 'yeast':
            dataset = YeastDataset(args.dataset_path, False)

            ndcgs = []
            for _, feat, edge_attr, edge_index, label in tqdm.tqdm(dataset, total=len(dataset)):
                feat = torch.tensor(feat, device=torch.device(args.device))
                edge_attr = torch.tensor(edge_attr, device=torch.device(args.device))
                edge_index = torch.tensor(edge_index, device=torch.device(args.device))
                label = torch.tensor(label, device=torch.device(args.device))
                ndcgs.append(ndcg_(model, feat, edge_attr, edge_index, label, NDCG_AT_K))

            metric, confidence = np.mean(ndcgs), 1.96 * np.std(ndcgs) / len(ndcgs) ** 0.5
        else:
            assert False

    print('{} +/- {}'.format(np.round(metric, 4), np.round(confidence, 4)))


if __name__ == '__main__':
    main()
