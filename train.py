import pytorch_lightning as pl
import torch
from torch.utils.data.dataset import IterableDataset
from torch import nn
import numpy as np
import pandas as pd
import tqdm

from dataset import EgoDataset, EgoLabelDataset
from model import WalkGNN
from utils import validate

DATA_PREFIX = '/home/e.zamyatin/walk_gnn/data/'
LIMIT = 1000

def make_submission(model, device, path):
    with open(path, 'w') as out:
        with torch.no_grad():
            model.eval()
            df = pd.read_csv(DATA_PREFIX + 'val_te_pr.csv')
            ego_ids = set(df[df['is_private']]['ego_id'])
            out.write('ego_id,u,v\n')
            for ego_id, ego_f, f, edge_index in tqdm.tqdm(EgoDataset(DATA_PREFIX + 'ego_net_te.csv', LIMIT), total=len(ego_ids)):
                if ego_id not in ego_ids: continue
                ego_f = ego_f.to(device)
                f = f.to(device)
                edge_index = edge_index.to(device).long()
                recs = model.recommend(ego_f, edge_index, f, 5)
                assert len(recs) == 5
                for u, v in recs:
                    out.write('{},{},{}\n'.format(ego_id, u, v))


def make_submission_and_validate(model, device, path):
    make_submission(model, device, path)
    print(validate(DATA_PREFIX + 'val_te_pr.csv', path, True))


class Trainer(WalkGNN):
    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks):
        super().__init__(node_dim, edge_dim, hid_dim, num_blocks)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def on_train_epoch_end(self):
        torch.save(self.state_dict(), DATA_PREFIX + 'wgnn_tiny_cut_{}.torch'.format(self.current_epoch))
        torch.onnx.export(self, (torch.rand((300, 8), device=torch.device(self.device)),
                                       torch.randint(1, 300, (2, 9000), device=torch.device(self.device)),
                                       torch.rand((9000, 4), device=torch.device(self.device))),
                          DATA_PREFIX + 'wgnn_tiny_cut_{}.onnx'.format(self.current_epoch),
                          export_params=True,
                          do_constant_folding=True,
                          dynamic_axes={'feat': {0: 'v_num'}, 'edge_index': {1: 'edge_num'},
                                        'edge_attr': {0: 'edge_num'}, 'output': {0: 'v_num', 1: 'v_num'}},
                          input_names=['feat', 'edge_index', 'edge_attr'], output_names=['output'], opset_version=13)
        make_submission_and_validate(self, self.device, DATA_PREFIX + 'submission_wgnn_tiny_cut_{}.csv'.format(self.current_epoch))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.parameters()), lr=0.0001)
        return optimizer


def main():
    model = Trainer(node_dim=8, edge_dim=4, hid_dim=8, num_blocks=6)
    ego_net_path = DATA_PREFIX + 'ego_net_tr.csv'
    label_path = DATA_PREFIX + 'label.csv'
    train_dataset = EgoLabelDataset(ego_net_path, label_path, LIMIT)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0)
    trainer = pl.Trainer(max_epochs=100, devices=[3], accelerator='gpu', accumulate_grad_batches=10)
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()

