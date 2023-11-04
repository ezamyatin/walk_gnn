import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataset import IterableDataset

from dataset import InMemoryEgoLabelDataset, DATA_PREFIX, LIMIT
from loss import PWLoss
from models import get_model
from validate import validate, NDCG_AT_K

TB_LOG_PATH = "/home/e.zamyatin/walk_gnn/tb_logs"


class LightningModel(LightningModule):
    def __init__(self, model, loss_obj, uuid):
        super().__init__()
        self.model = model
        self.uuid = uuid
        self.loss_obj = loss_obj

    def model_name(self):
        return self.model.__class__.__name__

    def training_step(self, batch, batch_idx):
        loss = self.loss_obj.compute_loss(self.model, batch, batch_idx)
        self.log("loss/train", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        torch.save(self.state_dict(), DATA_PREFIX + 'models/{}_{}_{}.torch'.format(self.model_name(), self.uuid, self.current_epoch))
        with torch.no_grad():
            metric, confidence = validate(self.model.eval(), DATA_PREFIX + "ego_net_te.csv", DATA_PREFIX + "val_te_pr.csv", NDCG_AT_K, False, self.device)
            print(metric)
            self.log("ndcg@{}/validation".format(NDCG_AT_K), metric, sync_dist=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.parameters()), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, verbose=True)
        return [optimizer], [scheduler]

    def get_logger(self):
        return TensorBoardLogger(
            TB_LOG_PATH,
            name="{}_{}".format(self.model_name(), self.uuid),
            default_hp_metric=False,
        )


def main():
    uuid = np.random.randint(1000000000)
    print("UUID:", uuid)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['walk_gnn', 'walk_gnn_no_attr', 'walk_gnn_no_node_attr', 'walk_gnn_no_edge_attr',
                                            'gine', 'gine_ohe', 'gin_ohe', 'gin_constant', 'rgine', 'gine_id_ohe',
                                            'ppgn', 'ppgn_no_attr'])

    parser.add_argument('--device', choices=['cpu'] + ['cuda:{}'.format(i) for i in range(4)])
    parser.add_argument('--state_dict_path', default=None)

    args = parser.parse_args()

    model = get_model(args)

    lit_model = LightningModel(model=model,
                               loss_obj=PWLoss(),
                               uuid=uuid)

    ego_net_path = DATA_PREFIX + 'ego_net_tr.csv'
    label_path = DATA_PREFIX + 'label.csv'
    train_dataset = InMemoryEgoLabelDataset(ego_net_path, label_path, LIMIT)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    trainer = pl.Trainer(max_epochs=100, devices=[int(args.device.split(":")[-1])], accelerator='gpu', accumulate_grad_batches=10, logger=[lit_model.get_logger()])
    trainer.fit(model=lit_model, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()

