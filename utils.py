import pandas as pd
import numpy as np


def validate(test_label_path, submission_path, private):
    label_df = pd.read_csv(test_label_path)
    if private:
        label_df = label_df[label_df['is_private']]
    else:
        label_df = label_df[~label_df['is_private']]
    subm_df = pd.read_csv(submission_path)
    assert len(subm_df.drop_duplicates(['ego_id', 'u', 'v'])) == len(subm_df)
    assert (subm_df.groupby('ego_id').count() > 5).sum().sum() == 0
    assert (label_df['u'] >= label_df['v']).sum() == 0
    subm_df['rank'] = (subm_df.groupby('ego_id').cumcount() + 1)
    df = pd.merge(label_df, subm_df, on=['ego_id', 'u', 'v'], how='left', indicator=True)
    df['hit'] = (df['_merge'] == 'both') * 1.0

    #if recall:
    #    return df.groupby('ego_id').mean()['hit'].mean()

    df['idcg'] = 1 / np.log2((df.groupby('ego_id').cumcount() + 1) + 1)
    df['dcg'] = 0
    df.loc[~df['rank'].isna(), 'dcg'] = 1 / np.log2(df['rank'] + 1)
    grouped_df = df.groupby('ego_id').sum()
    result = (grouped_df['dcg'] / grouped_df['idcg']).mean()
    return result