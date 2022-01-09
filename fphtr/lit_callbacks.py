import math
from typing import Tuple, Optional

from fphtr.util import matplotlib_imshow

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback


class LogWorstPredictions(Callback):
    """
    At the end of every epoch, log the predictions with the highest loss values,
    i.e. the worst predictions of the model.
    """

    def __init__(self):
        pass

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if pl_module.all_logits is None or pl_module.all_targets is None:
            ...
        # TODO


class LogModelPredictions(Callback):
    """
    Use a fixed test batch to monitor model predictions at the end of every epoch.

    Specifically: it generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's prediction alongside the actual target.
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        val_batch: Tuple[torch.Tensor, torch.Tensor],
        use_gpu: bool = True,
        data_format: str = "word",
        train_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.label_encoder = label_encoder
        self.val_batch = val_batch
        self.use_gpu = use_gpu
        self.data_format = data_format
        self.train_batch = train_batch

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        self._predict_intermediate(trainer, pl_module, split="val")

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if self.train_batch is not None:
            self._predict_intermediate(trainer, pl_module, split="train")

    def _predict_intermediate(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", split="val"
    ):
        """Make predictions on a fixed batch of data and log the results to Tensorboard."""

        # Make predictions.
        if split == "train":
            imgs, targets = self.train_batch
        else:  # split == "val"
            imgs, targets = self.val_batch
        with torch.inference_mode():
            pl_module.eval()
            _, preds, _ = pl_module(imgs.cuda() if self.use_gpu else imgs)

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
            ncols = 2 if self.data_format == "word" else 1
            nrows = math.ceil(preds.size(0) / ncols)
            ax = fig.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
            matplotlib_imshow(imgs[i])
            ax.set_title(f"Pred: {pred_str}\nTarget: {target_str}")

        # Log the results to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(
            f"{split}: predictions vs targets", fig, trainer.global_step
        )
        plt.close(fig)
