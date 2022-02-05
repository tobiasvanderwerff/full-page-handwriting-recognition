from typing import Optional, Dict, Union

from models import FullPageHTREncoderDecoder
from util import LabelEncoder

import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch import Tensor


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
        learning_rate: float = 0.0002,
        label_smoothing: float = 0.0,
        max_seq_len: int = 500,
        d_model: int = 260,
        num_layers: int = 6,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        encoder_name: str = "resnet18",
        drop_enc: int = 0.1,
        drop_dec: int = 0.1,
        activ_dec: str = "gelu",
        vocab_len: Optional[int] = None,  # if not specified len(label_encoder) is used
        params_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        # Save hyperparameters.
        self.learning_rate = learning_rate
        if params_to_log is not None:
            self.save_hyperparameters(params_to_log)
        self.save_hyperparameters(
            "learning_rate",
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
            vocab_len=vocab_len,
            label_smoothing=label_smoothing,
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
        return self.val_or_test_step(batch)

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(batch)

    def val_or_test_step(self, batch) -> Tensor:
        imgs, targets = batch
        logits, _, loss = self(imgs, targets)
        _, preds = logits.max(-1)

        # Update and log metrics.
        self.model.cer_metric(preds, targets)
        self.model.wer_metric(preds, targets)
        self.log("char_error_rate", self.model.cer_metric, prog_bar=True)
        self.log("word_error_rate", self.model.wer_metric, prog_bar=True)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # fmt: off
        parser = parent_parser.add_argument_group("LitFullPageHTREncoderDecoder")
        parser.add_argument("--learning_rate", type=float, default=0.0002)
        parser.add_argument("--encoder", type=str, default="resnet18",
                            choices=["resnet18", "resnet34", "resnet50"])
        parser.add_argument("--d_model", type=int, default=260)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--nhead", type=int, default=4)
        parser.add_argument("--dim_feedforward", type=int, default=1024)
        parser.add_argument("--drop_enc", type=float, default=0.1,
                            help="Encoder dropout.")
        parser.add_argument("--drop_dec", type=float, default=0.1,
                            help="Decoder dropout.")
        return parent_parser
        # fmt: on
