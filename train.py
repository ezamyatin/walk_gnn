import pytorch_lightning as pl
import torch
from torch.utils.data.dataset import IterableDataset
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import numpy as np
import pandas as pd
import tqdm

from dataset import EgoDataset, InMemoryEgoLabelDataset, DATA_PREFIX, LIMIT
from models.walk_gnn import WalkGNN
from validate import validate, NDCG_AT_K


class Trainer(WalkGNN):
    def __init__(self, node_dim, edge_dim, hid_dim, num_blocks, uuid):
        super().__init__(node_dim, edge_dim, hid_dim, num_blocks)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.uuid = uuid

    def on_train_epoch_end(self):
        torch.save(self.state_dict(), DATA_PREFIX + 'models/wgnn_tiny_cut_{}_{}.torch'.format(self.uuid, self.current_epoch))
        with torch.no_grad():
            metric = validate(self.eval(), DATA_PREFIX + "ego_net_te.csv", DATA_PREFIX + "val_te_pr.csv", NDCG_AT_K, False)
            print(metric)
            self.log("ndcg@{}/validation".format(NDCG_AT_K), metric, sync_dist=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.parameters()), lr=0.0001)
        return optimizer


def main():
    uuid = np.random.randint(1000000000)
    model = Trainer(node_dim=8, edge_dim=4, hid_dim=8, num_blocks=6, uuid=uuid)
    ego_net_path = DATA_PREFIX + 'ego_net_tr.csv'
    label_path = DATA_PREFIX + 'label.csv'
    train_dataset = InMemoryEgoLabelDataset(ego_net_path, label_path, LIMIT)
    print("UUID:", uuid)

    logger = TensorBoardLogger(
        "/home/e.zamyatin/walk_gnn/tb_logs",
        name="walk_gnn_{}".format(uuid),
        default_hp_metric=False,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    trainer = pl.Trainer(max_epochs=100, devices=[2], accelerator='gpu', accumulate_grad_batches=10, logger=[logger])
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()

