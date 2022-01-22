import math
from typing import Tuple, Optional

from lit_models import LitFullPageHTREncoderDecoder
from util import matplotlib_imshow, LabelEncoder, decode_prediction_and_target
from data import IAMDataset

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


class LogWorstPredictions(Callback):
    """
    At the end of training, log the worst image prediction, meaning the predictions
    with the highest character error rates.
    """

    def __init__(
        self,
        val_dataloader: DataLoader,
        val_only: bool = False,
        data_format: str = "word",
        worst_k: int = 10,
    ):
        self.val_dataloader = val_dataloader
        self.val_only = val_only
        self.data_format = data_format
        self.worst_k = worst_k

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.val_only:
            # This is to make sure the callback is called when model training
            # is skipped.
            self.log_worst_predictions(trainer, pl_module)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.log_worst_predictions(trainer, pl_module)

    def log_worst_predictions(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        img_cers = []
        device = "cuda:0" if pl_module.on_gpu else "cpu"
        if not self.val_only:
            self._load_best_model(trainer, pl_module)
            pl_module = trainer.model

        print("Running validation inference on best model...")

        # Run inference on the validation set.
        pl_module.eval()
        for img, target in self.val_dataloader:
            assert target.ndim == 2, target.ndim
            cer_metric = pl_module.model.cer_metric
            with torch.inference_mode():
                logits, preds, _ = pl_module(img.to(device), target.to(device))
                for prd, tgt, im in zip(preds, target, img):
                    cer_metric.reset()
                    cer = cer_metric(prd.unsqueeze(0), tgt.unsqueeze(0))
                    img_cers.append((im, cer, prd, tgt))
            break

        # Log the worst k predictions.
        img_cers.sort(key=lambda x: x[1], reverse=True)  # sort by CER
        img_cers = img_cers[: self.worst_k]
        fig = plt.figure(figsize=(12, 16))
        for i, (im, cer, prd, tgt) in enumerate(img_cers):
            pred_str, target_str = decode_prediction_and_target(
                prd, tgt, pl_module.model.label_encoder, pl_module.decoder.eos_tkn_idx
            )

            # Create plot.
            ncols = 2 if self.data_format == "word" else 1
            nrows = math.ceil(self.worst_k / ncols)
            ax = fig.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
            matplotlib_imshow(im, IAMDataset.MEAN, IAMDataset.STD)
            ax.set_title(f"Pred: {pred_str} (CER: {cer:.2f})\nTarget: {target_str}")

        # Log the results to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(f"val: worst predictions", fig, trainer.global_step)
        plt.close(fig)

    @staticmethod
    def _load_best_model(trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        ckpt_callback = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_callback = cb
                break
        assert ckpt_callback is not None, "ModelCheckpoint not found in callbacks."
        best_model_path = ckpt_callback.best_model_path

        print(f"Loading best model at {best_model_path}")
        label_encoder = pl_module.model.label_encoder
        model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
            best_model_path,
            label_encoder=label_encoder,
        )
        trainer.model = model


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

        # Decode predictions and generate a plot.
        fig = plt.figure(figsize=(12, 16))
        for i, (p, t) in enumerate(zip(preds, targets)):
            pred_str, target_str = decode_prediction_and_target(
                p, t, self.label_encoder, pl_module.decoder.eos_tkn_idx
            )

            # Create plot.
            ncols = 2 if self.data_format == "word" else 1
            nrows = math.ceil(preds.size(0) / ncols)
            ax = fig.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
            matplotlib_imshow(imgs[i], IAMDataset.MEAN, IAMDataset.STD)
            ax.set_title(f"Pred: {pred_str}\nTarget: {target_str}")

        # Log the results to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(
            f"{split}: predictions vs targets", fig, trainer.global_step
        )
        plt.close(fig)
