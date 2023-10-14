import numpy as np
import pandas as pd
import torch
import tqdm
from dataset import EgoDataset, DATA_PREFIX, LIMIT
import argparse

from models.heuristic import AdamicAdar, WeightedAdamicAdar
from models.walk_gnn import WalkGNN

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
    result = (grouped_df['dcg'] / grouped_df['idcg']).mean()
    return result


def validate(model, test_ego_path, test_label_path, k, private):

    label_df = pd.read_csv(test_label_path)
    ego_ids = set(label_df[label_df['is_private'] == private]['ego_id'])

    out_df = {
        'ego_id': [],
        'u': [],
        'v': [],
    }

    for ego_id, ego_f, f, edge_index in tqdm.tqdm(EgoDataset(test_ego_path, LIMIT), total=len(ego_ids) * 2):
        if ego_id not in ego_ids: continue
        recs = model.recommend(ego_f, edge_index, f, k)
        assert len(recs) == k
        for u, v in recs:
            out_df['ego_id'].append(ego_id)
            out_df['u'].append(u)
            out_df['v'].append(v)
    print()
    return ndcg_at_k(label_df, pd.DataFrame.from_dict(out_df), k, private)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['aa', 'walk_gnn'])
    parser.add_argument('--state_dict_path', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    model = None
    if args.model == 'walk_gnn':
        model = WalkGNN(node_dim=8, edge_dim=4, hid_dim=8, num_blocks=6)
        model.load_state_dict(torch.load(args.state_dict_path))
        if args.device is not None:
            model.to(args.device)
    elif args.model == 'aa':
        model = AdamicAdar()
    elif args.model == 'waa':
        model = WeightedAdamicAdar()

    metric = validate(model, DATA_PREFIX + "ego_net_te.csv", DATA_PREFIX + "val_te_pr.csv", NDCG_AT_K, True)
    print(metric)


if __name__ == '__main__':
    main()
