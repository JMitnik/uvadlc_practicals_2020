from pickletools import optimize
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn

class CharacterPredictor(pl.LightningModule):

    def __init__(self, model, device) -> None:
        super().__init__()
        self.model = model
        self.active_device = device

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, idx):
        X, y = batch
        X = torch.stack(X, 1).to(self.device)
        y = torch.stack(y, 1).to(self.device)
        preds = self.model(X)

        # Flatten preds and y
        preds = preds.reshape(-1, preds.shape[2])
        y = y.reshape(-1)

        losses_mean = F.cross_entropy(preds, y)
        self.log('loss', losses_mean)

        acc = 0
        return losses_mean, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer