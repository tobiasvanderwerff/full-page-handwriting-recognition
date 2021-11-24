from typing import Callable, Optional, Dict, Union, Any

from models import FullPageHTREncoder, FullPageHTRDecoder

from metrics import CharacterErrorRate, WordErrorRate

import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch import Tensor
from sklearn.preprocessing import LabelEncoder


class LitFullPageHTREncoderDecoder(pl.LightningModule):
    encoder: FullPageHTREncoder
    decoder: FullPageHTRDecoder
    cer_metric: CharacterErrorRate
    wer_metric: WordErrorRate
    loss_fn: Callable

    def __init__(
        self,
        label_encoder: LabelEncoder,
        encoder_name: str,
        vocab_len: int,
        d_model: int,
        max_seq_len: int,
        eos_tkn_idx: int,
        sos_tkn_idx: int,
        pad_tkn_idx: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        drop_enc: int = 0.1,
        drop_dec: int = 0.5,
        activ_dec: str = "gelu",
        params_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
    ):
        super().__init__()

        # Save hyperparameters.
        opt_params = self.full_page_htr_optimizer_params()
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

        # Initialize models.
        self.encoder = FullPageHTREncoder(
            d_model, model_name=encoder_name, dropout=drop_enc
        )
        self.decoder = FullPageHTRDecoder(
            vocab_len=vocab_len,
            max_seq_len=max_seq_len,
            eos_tkn_idx=eos_tkn_idx,
            sos_tkn_idx=sos_tkn_idx,
            pad_tkn_idx=pad_tkn_idx,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=drop_dec,
            activation=activ_dec,
        )

        # Initialize metrics and loss function.
        self.cer_metric = CharacterErrorRate(label_encoder)
        self.wer_metric = WordErrorRate(label_encoder)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.decoder.pad_tkn_idx)

    def forward(self, imgs: Tensor):
        logits, sampled_ids = self.decoder(self.encoder(imgs))
        return logits, sampled_ids

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        memory = self.encoder(imgs)
        logits = self.decoder.decode_teacher_forcing(memory, targets)

        loss = self.loss_fn(logits.transpose(1, 2), targets)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits, _ = self(imgs)
        _, preds = logits.max(-1)

        # Calculate metrics and loss.
        cer = self.cer_metric(preds, targets)
        wer = self.wer_metric(preds, targets)
        loss = self.loss_fn(
            logits[:, : targets.size(1), :].transpose(1, 2),
            targets[:, : logits.size(1)],
        )
        # Log metrics and loss.
        self.log("char_error_rate", cer, prog_bar=True)
        self.log("word_error_rate", wer)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log(
            "hp_metric", wer
        )  # this will show up in the Tensorboard hparams tab, used for comparing different models

        return loss

    def configure_optimizers(self):
        # By default use the optimizer parameters specified in Singh et al. (2021).
        params = self.full_page_htr_optimizer_params()
        optimizer_name = params.pop("optimizer_name")
        optimizer = getattr(optim, optimizer_name)(self.parameters(), **params)
        return optimizer

    @staticmethod
    def full_page_htr_optimizer_params() -> Dict[str, Any]:
        """See Singh et al, page 9."""
        return {"optimizer_name": "AdamW", "lr": 0.0002, "betas": (0.9, 0.999)}

    def get_progress_bar_dict(self):
        # Don't show the version number.
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
