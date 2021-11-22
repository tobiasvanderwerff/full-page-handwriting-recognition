from typing import Callable, Optional, Dict, Union, Any

from models import FullPageHTREncoder, FullPageHTRDecoder
from metrics import CharacterErrorRate

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor


class LitFullPageHTREncoderDecoder(pl.LightningModule):
    encoder: FullPageHTREncoder
    decoder: FullPageHTRDecoder
    cer_metric: CharacterErrorRate
    # wer_metric: WordErrorRate
    loss_fn: Callable

    def __init__(
        self,
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
        self.cer_metric = CharacterErrorRate()
        # self.wer_metric = WordErrorRate()
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
        B, T = targets.shape

        logits, _ = self(imgs)
        _, preds = logits.max(-1)
        # self.last_preds = preds.detach()  # for callbacks
        if logits.size(1) < targets.size(1):
            # Pad logits until maximum target length.
            _, _, n_classes = logits.shape
            pad = F.one_hot(torch.tensor(self.decoder.pad_tkn_idx), n_classes)
            pad = pad.expand(B, T - logits.size(1), -1).to(self.device)
            logits = torch.cat([logits, pad], 1)

        loss = self.loss_fn(logits[:, :T, :].transpose(1, 2), targets)
        cer = self.cer_metric(logits[:, :T, :].argmax(-1), targets)
        # wer = self.wer_metric(logits[:, :T, :].argmax(-1), targets)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("char_error_rate", cer)
        # self.log("word_error_rate", wer)
        self.log(
            "hp_metric", cer
        )  # this will show up in the Tensorboard hparams tab, to compare different models

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