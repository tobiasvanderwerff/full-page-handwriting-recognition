import math
from typing import Tuple, Optional

import torch

from util import matplotlib_imshow

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import Callback
from sklearn.preprocessing import LabelEncoder


class LogModelPredictions(Callback):
    """
    Use a fixed test batch to monitor model predictions at the end of every epoch.

    Specifically: it generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's prediction alongside the actual target.
    """

    def __init__(
        self,
        label_encoder: "LabelEncoder",
        val_batch: Tuple[torch.Tensor, torch.Tensor],
        use_gpu: bool = True,
        train_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.label_encoder = label_encoder
        self.val_batch = val_batch
        self.use_gpu = use_gpu
        self.train_batch = train_batch

    def on_validation_epoch_end(self, trainer, pl_module: "LightningModule"):
        self._predict_intermediate(trainer, pl_module, split="val")

    def on_train_epoch_end(self, trainer, pl_module: "LightningModule"):
        if self.train_batch is not None:
            self._predict_intermediate(trainer, pl_module, split="train")

    def _predict_intermediate(self, trainer, pl_module: "LightningModule", split="val"):
        """Make predictions on a fixed batch of data and log the results to Tensorboard."""

        # Make predictions.
        if split == "train":
            imgs, targets = self.train_batch
        else:  # split == "val"
            imgs, targets = self.val_batch
        with torch.no_grad():
            pl_module.eval()  # TODO: check to what value this should be set afterwards
            _, preds = pl_module(imgs.cuda() if self.use_gpu else imgs)

        # Find padding and <EOS> positions in predictions and targets.
        eos_idxs_pred = (
            (preds == pl_module.decoder.eos_tkn_idx).float().argmax(1).tolist()
        )
        eos_idxs_tgt = (
            (targets == pl_module.decoder.eos_tkn_idx).float().argmax(1).tolist()
        )

        # Decode predictions and generate a plot.
        fig = plt.figure(figsize=(12, 16))
        for i, (p, t) in enumerate(zip(preds.tolist(), targets.tolist())):
            # Decode predictions and targets.
            max_pred_idx, max_target_idx = eos_idxs_pred[i], eos_idxs_tgt[i]
            p = p[1:]  # skip the initial <SOS> token, which is added by default
            if max_pred_idx != 0:
                pred_str = "".join(
                    self.label_encoder.inverse_transform(p)[:max_pred_idx]
                )
            else:
                pred_str = "".join(self.label_encoder.inverse_transform(p))
            if max_target_idx != 0:
                target_str = "".join(
                    self.label_encoder.inverse_transform(t)[:max_target_idx]
                )
            else:
                target_str = "".join(self.label_encoder.inverse_transform(t))

            # Create plot.
            ax = fig.add_subplot(
                math.ceil(preds.size(0) / 2), 2, i + 1, xticks=[], yticks=[]
            )
            matplotlib_imshow(imgs[i])
            ax.set_title(f"Pred: {pred_str}\nTarget: {target_str}")

        # Log the results to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(
            f"{split}: predictions vs targets", fig, trainer.global_step
        )
