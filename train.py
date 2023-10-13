import pytorch_lightning as pl
import torch
from torch.utils.data.dataset import IterableDataset
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import numpy as np
import pandas as pd
import tqdm

from dataset import EgoDataset, EgoLabelDataset
from model import WalkGNN
from utils import validate

DATA_PREFIX = '/home/e.zamyatin/walk_gnn/data/'
LIMIT = None


def make_submission(model, device, path):
    with open(path, 'w') as out:
        with torch.no_grad():
            model.eval()
            df = pd.read_csv(DATA_PREFIX + 'val_te_pr.csv')
            ego_ids = set(df[df['is_private']]['ego_id'])
            out.write('ego_id,u,v\n')
            for ego_id, ego_f, f, edge_index in tqdm.tqdm(EgoDataset(DATA_PREFIX + 'ego_net_te.csv', LIMIT), total=len(ego_ids) * 2):
                if ego_id not in ego_ids: continue
                ego_f = torch.Tensor(ego_f).to(device)
                f = torch.Tensor(f).to(device)
                edge_index = torch.Tensor(edge_index).to(device).long()
                recs = model.recommend(ego_f, edge_index, f, 5)
                assert len(recs) == 5
                for u, v in recs:
                    out.write('{},{},{}\n'.format(ego_id, u, v))


def make_submission_and_validate(model, device, path):
    make_submission(model, device, path)
    print()
    r = validate(DATA_PREFIX + 'val_te_pr.csv', path, True)
    print(r)
    model.log("ndcg@5/validation", r, sync_dist=True, on_epoch=True)


class Trainer(WalkGNN):
    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks, uuid):
        super().__init__(node_dim, edge_dim, hid_dim, num_blocks)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.uuid = uuid

    def on_train_epoch_end(self):
        torch.save(self.state_dict(), DATA_PREFIX + 'models/wgnn_tiny_cut_{}_{}.torch'.format(self.uuid, self.current_epoch))
        make_submission_and_validate(self, self.device, DATA_PREFIX + 'submissions/submission_wgnn_tiny_cut_{}_{}.csv'.format(self.uuid, self.current_epoch))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.parameters()), lr=0.0001)
        return optimizer


def main():
    uuid = np.random.randint(1000000000)
    model = Trainer(node_dim=8, edge_dim=4, hid_dim=8, num_blocks=6, uuid=uuid)
    ego_net_path = DATA_PREFIX + 'ego_net_tr.csv'
    label_path = DATA_PREFIX + 'label.csv'
    train_dataset = EgoLabelDataset(ego_net_path, label_path, LIMIT)
    print("UUID:", uuid)

    logger = TensorBoardLogger(
        "/home/e.zamyatin/walk_gnn/tb_logs",
        name="walk_gnn_{}".format(uuid),
        default_hp_metric=False,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    trainer = pl.Trainer(max_epochs=100, devices=[3], accelerator='gpu', accumulate_grad_batches=10, logger=[logger])
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()

