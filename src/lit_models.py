from typing import Optional, Dict, Union

from models import FullPageHTREncoderDecoder

import torch.optim as optim
import pytorch_lightning as pl
from torch import Tensor
from sklearn.preprocessing import LabelEncoder


class LitFullPageHTREncoderDecoder(pl.LightningModule):
    model: FullPageHTREncoderDecoder

    """
    Pytorch Lightning module that acting as a wrapper around the
    FullPageHTREncoderDecoder class.

    Using a PL module allows the model to be used in conjunction with a Pytorch
    Lightning Trainer, and takes care of logging relevant metrics to Tensorboard.
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        max_seq_len: int = 500,
        d_model: int = 260,
        num_layers: int = 6,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        encoder_name: str = "resnet18",
        drop_enc: int = 0.5,
        drop_dec: int = 0.5,
        activ_dec: str = "gelu",
        params_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        # Save hyperparameters.
        opt_params = FullPageHTREncoderDecoder.full_page_htr_optimizer_params()
        if params_to_log is not None:
            self.save_hyperparameters(params_to_log)
        self.save_hyperparameters(opt_params)
        self.save_hyperparameters(
            "d_model",
            "num_layers",
            "nhead",
            "dim_feedforward",
            "max_seq_len",
            "encoder_name",
            "drop_enc",
            "drop_dec",
            "activ_dec",
        )

        # Initialize the model.
        self.model = FullPageHTREncoderDecoder(
            label_encoder=label_encoder,
            max_seq_len=max_seq_len,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            encoder_name=encoder_name,
            drop_enc=drop_enc,
            drop_dec=drop_dec,
            activ_dec=activ_dec,
        )

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None):
        return self.model(imgs, targets)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits, loss = self.model.forward_teacher_forcing(imgs, targets)
        self.log("train_loss", loss, sync_dist=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch

        # Forward pass.
        logits, _, loss = self(imgs, targets)
        _, preds = logits.max(-1)

        # Calculate metrics.
        metrics = self.model.calculate_metrics(preds, targets)

        # Log metrics and loss.
        self.log("char_error_rate", metrics["char_error_rate"], prog_bar=True)
        self.log("word_error_rate", metrics["word_error_rate"], prog_bar=True)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        # `hp_metric` will show up in the Tensorboard hparams tab, used for comparing
        # different models.
        self.log("hp_metric", metrics["char_error_rate"])

        return loss

    def configure_optimizers(self):
        # By default use the optimizer parameters specified in Singh et al. (2021).
        params = FullPageHTREncoderDecoder.full_page_htr_optimizer_params()
        optimizer_name = params.pop("optimizer_name")
        optimizer = getattr(optim, optimizer_name)(self.parameters(), **params)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitFullPageHTREncoderDecoder")
        parser.add_argument(
            "--encoder",
            type=str,
            choices=["resnet18", "resnet34", "resnet50"],
            default="resnet18",
        )
        parser.add_argument("--d_model", type=int, default=260)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--nhead", type=int, default=4)
        parser.add_argument("--dim_feedforward", type=int, default=1024)
        parser.add_argument(
            "--drop_enc", type=float, default=0.5, help="Encoder dropout."
        )
        parser.add_argument(
            "--drop_dec", type=float, default=0.5, help="Decoder dropout."
        )
        return parent_parser
