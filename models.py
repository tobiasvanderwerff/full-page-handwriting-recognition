"""
Symbol table:
    B: batch size
    T: target sequence length
    w: image width
    h: image height
"""

import math
from typing import List, Callable

from pl_callbacks import ShowPredictions

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch import Tensor
import pytorch_lightning as pl


# __all__ = [FullPageHTREncoderDecoder, FullPageHTREncoder, FullPageHTRDecoder]


class PositionalEmbedding1D(nn.Module):
    """
    Implements 1D sinusoidal embeddings.

    Adapted from 'The Annotated Transformer', http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros((max_len, d_model), requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add a 1D positional embedding to an input tensor.

        Args:
            x (Tensor): tensor of shape (B, T, d_model) to add positional
                embedding to
        """
        _, T, _ = x.shape
        return x + self.pe[:, :T]


class PositionalEmbedding2D(nn.Module):
    """Implements 2D sinusoidal embeddings. See p.7 of Singh et al. for more details."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        assert d_model % 4 == 0

        # Calculate all positional embeddings once and save them for re-use.
        pe_x = torch.zeros((max_len, d_model // 2), requires_grad=False)
        pe_y = torch.zeros((max_len, d_model // 2), requires_grad=False)

        pos = torch.arange(0, max_len).unsqueeze(1)
        # Div term term is calculated in log space, as done in other implementations;
        # this is most likely for numerical stability/precision. The expression below is
        # equivalent to:
        #     div_term = 10000 ** (torch.arange(0, d_model // 2, 2) / d_model)
        div_term = torch.exp(
            -math.log(10000) * torch.arange(0, d_model // 2, 2) / d_model
        )
        pe_y[:, 0::2] = torch.sin(pos * div_term)  # shape: (max_len, d_model/4)
        pe_y[:, 1::2] = torch.cos(pos * div_term)
        pe_x[:, 0::2] = torch.sin(pos * div_term)
        pe_x[:, 1::2] = torch.cos(pos * div_term)

        # Include positional encoding into module's state, to be saved alongside parameters.
        self.register_buffer("pe_x", pe_x)
        self.register_buffer("pe_y", pe_y)

    def forward(self, x):
        """
        Add a 2D positional embedding to an input tensor.

        Args:
            x (Tensor): tensor of shape (B, w, h, d_model) to add positional
                embedding to
        """
        _, w, h, _ = x.shape

        pe_x_ = self.pe_x[:w, :].unsqueeze(1).expand(-1, h, -1)  # (w, h, d_model/2)
        pe_y_ = self.pe_y[:h, :].unsqueeze(0).expand(w, -1, -1)  # (w, h, d_model/2)
        pe = torch.cat([pe_y_, pe_x_], -1)  # (w, h, d_model)
        pe = pe.unsqueeze(0)  # (1, w, h, d_model)
        return x + pe


class FullPageHTRDecoder(nn.Module):
    decoder: nn.TransformerDecoder
    clf: nn.Linear
    emb: nn.Embedding
    pos_emb: PositionalEmbedding1D
    drop: nn.Dropout

    vocab: List[str]
    max_seq_len: int
    eos_tkn_idx: int
    sos_tkn_idx: int
    pad_tkn_idx: int
    d_model: int
    num_layers: int
    nhead: int
    dim_feedforward: int
    dropout: float
    activation: str

    def __init__(
        self,
        vocab: List[str],
        max_seq_len: int,
        eos_tkn_idx: int,
        sos_tkn_idx: int,
        pad_tkn_idx: int,
        d_model: int = 260,
        num_layers: int = 6,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.5,
        activation: str = "gelu",
    ):
        super().__init__()
        assert d_model % 4 == 0

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.eos_tkn_idx = eos_tkn_idx
        self.sos_tkn_idx = sos_tkn_idx
        self.pad_tkn_idx = pad_tkn_idx
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        self.emb = nn.Embedding(len(vocab), d_model)
        self.pos_emb = PositionalEmbedding1D(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.clf = nn.Linear(d_model, len(vocab))
        self.drop = nn.Dropout(p=dropout)

    def forward(self, memory: Tensor):
        """Greedy decoding."""
        B, _, _ = memory.shape
        all_logits = []
        sampled_ids = [torch.full([B], self.sos_tkn_idx).to(memory.device)]
        tgt = self.pos_emb(  # tgt: (bs, 1, d_model)
            self.emb(sampled_ids[0]).unsqueeze(1) * math.sqrt(self.d_model)
        )
        tgt = self.drop(tgt)
        for t in range(self.max_seq_len):
            tgt_mask = self.subsequent_mask(len(sampled_ids)).to(memory.device)
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # out: (B, T, d_model)
            logits = self.clf(out[:, -1, :])  # logits: (B, vocab_size)
            _, pred = torch.max(logits, -1)
            all_logits.append(logits)
            sampled_ids.append(pred)
            if (pred == self.eos_tkn_idx).all():
                break
            tgt_ext = self.drop(
                self.pos_emb.pe[:, len(sampled_ids)]
                + self.emb(pred) * math.sqrt(self.d_model)
            ).unsqueeze(1)
            tgt = torch.cat([tgt, tgt_ext], 1)
        sampled_ids = torch.stack(sampled_ids, 1)
        all_logits = torch.stack(all_logits, 1)

        # Replace all sampled_ids tokens after <EOS> with <PAD> tokens.
        eos_idxs = (sampled_ids == self.eos_tkn_idx).float().argmax(1)
        for i in range(B):
            sampled_ids[i, : eos_idxs[i] + 1] = self.pad_tkn_idx

        return all_logits, sampled_ids

    def decode_teacher_forcing(self, memory: Tensor, tgt: Tensor):
        """
        Args:
            memory (Tensor): tensor of shape (B, w*h, d_model), containing encoder
                output, used as Key and Value vectors for the decoder
            tgt (Tensor): tensor of shape (B, T), containing the targets used for
                teacher forcing
        Returns:
            logits Tensor of shape (B, T, vocab_len)
        """
        B, T = tgt.shape

        # Shift the elements of tgt to the right.
        tgt = torch.cat(
            [
                torch.full([B], self.sos_tkn_idx).unsqueeze(1).to(memory.device),
                tgt[:, :-1],
            ],
            1,
        )

        # Masking for causal self-attention. This is a combination of pad token masking
        # (tgt_key_padding_mask) and causal self-attention masking (tgt_mask), where the
        # tgt_mask is of shape (T, T) where we shift the targets to the right
        # by one.
        tgt_key_padding_mask = tgt == self.pad_tkn_idx
        tgt_mask = self.subsequent_mask(T).to(tgt.device)

        tgt = self.pos_emb(self.emb(tgt) * math.sqrt(self.d_model))
        tgt = self.drop(tgt)
        out = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        logits = self.clf(out)
        return logits

    @staticmethod
    def subsequent_mask(size: int):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 1


class FullPageHTREncoder(nn.Module):
    """
    Image encoder used in Singh et al.

    Note: in the paper the following encoders are tried: resnet18, resnet34, resnet50.
    """

    encoder: nn.Sequential
    linear: nn.Conv2d
    pos_emb: PositionalEmbedding2D
    drop: nn.Dropout
    d_model: int
    model_name: str

    # TODO: default dropout rate is taken from the original Transformers paper, because
    # it was not provided in Singh et al. However, the decoder in Singh et al. uses
    # dropout=0.5, which makes me think perhaps the dropout rate for the encoder should
    # be higher.

    def __init__(self, d_model, model_name="resnet18", dropout=0.1):
        super().__init__()

        assert d_model % 4 == 0
        # TODO: figure out what to set max_len to, by looking at the final feature map
        # dimensions h and w of the resnet encoder. Then max_len is max(h, w).
        _models = ["resnet18", "resnet34", "resnet50"]
        err_message = f"{model_name} is not an available option: {_models}"
        assert model_name in _models, err_message

        self.d_model = d_model
        self.model_name = model_name
        self.pos_emb = PositionalEmbedding2D(d_model, max_len=500)
        self.drop = nn.Dropout(p=dropout)

        resnet = getattr(torchvision.models, model_name)(pretrained=False)
        modules = list(resnet.children())

        # Change the first conv layer to take as input a single channel image.
        cnv_1 = modules[0]
        cnv_1 = nn.Conv2d(
            1,
            cnv_1.out_channels,
            cnv_1.kernel_size,
            cnv_1.stride,
            cnv_1.padding,
            bias=cnv_1.bias,
        )
        self.encoder = nn.Sequential(cnv_1, *modules[1:-2])
        self.linear = nn.Conv2d(
            resnet.fc.in_features, d_model, kernel_size=1
        )  # 1x1 convolution
        # self.bn = nn.BatchNorm2d(d_model)

    def forward(self, imgs):
        x = self.encoder(imgs.unsqueeze(1))  # x: (B, d_model, w, h)
        x = self.linear(x).transpose(1, 2).transpose(2, 3)  # x: (B, w, h, d_model)
        # x = self.bn(x.transpose(2, 3).transpose(1, 2)).transpose(1, 2).transpose(2, 3)  # x: (B, w, h, d_model)
        x = self.pos_emb(x)  # x: (B, w, h, d_model)
        x = self.drop(x)  # x: (B, w, h, d_model)
        x = x.flatten(1, 2)  # x: (B, w*h, d_model)
        return x


class FullPageHTREncoderDecoder(pl.LightningModule):
    """
    Implementation of model specified in "Full Page Handwriting Recognition
    via Image to Sequence Extraction" by Singh et al.

    All hyperparameters are based on training details specified in the paper,
    whenever they were available.
    """

    encoder: FullPageHTREncoder
    decoder: FullPageHTRDecoder
    loss_fn: Callable

    def __init__(self, encoder: FullPageHTREncoder, decoder: FullPageHTRDecoder):
        super().__init__()
        assert encoder.d_model == decoder.d_model

        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=decoder.pad_tkn_idx)

    def forward(self, imgs: Tensor):
        logits, sampled_ids = self.decode(self.encode(imgs))
        return logits, sampled_ids

    def encode(self, imgs: Tensor):
        return self.encoder(imgs)

    def decode(self, memory: Tensor):
        return self.decoder(memory)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        memory = self.encoder(imgs)
        logits = self.decoder.decode_teacher_forcing(memory, targets)

        self._logits, self._targets = logits, targets  # for callbacks

        loss = self.loss_fn(logits.transpose(1, 2), targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        B, T = targets.shape
        logits, _ = self.forward(inputs)
        self._logits, self._targets = logits, targets  # for callbacks
        if logits.size(1) < targets.size(1):
            # Pad logits until maximum target length.
            _, _, n_classes = logits.shape
            pad = F.one_hot(torch.tensor(self.decoder.pad_tkn_idx), n_classes)
            pad = pad.expand(B, T - logits.size(1), -1).to(logits.device)
            logits = torch.cat([logits, pad], 1)
        loss = self.loss_fn(logits[:, :T, :].transpose(1, 2), targets)
        return loss

    def configure_optimizers(self):
        # See Singh et al, page 9.
        optimizer = optim.AdamW(self.parameters(), lr=0.0002, betas=(0.9, 0.999))
        return optimizer
