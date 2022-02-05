"""
Implementation of model specified in "Full Page Handwriting Recognition
via Image to Sequence Extraction" by Singh et al.
"""

import math
from typing import Dict, Any, Tuple, Optional, Union, Callable

from metrics import CharacterErrorRate, WordErrorRate
from util import LabelEncoder

import torch
import torch.nn as nn
import torchvision
from torch import Tensor


class PositionalEmbedding1D(nn.Module):
    """
    Implements 1D sinusoidal embeddings.

    Adapted from 'The Annotated Transformer', http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, d_model, max_len=1000):
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
        # assert T <= self.pe.size(0) \
        assert T <= self.pe.size(1), (
            f"Stored 1D positional embedding does not have enough dimensions for the current feature map. "
            f"Currently max_len={self.pe.size(1)}, T={T}. Consider increasing max_len such that max_len >= T."
        )
        return x + self.pe[:, :T]


class PositionalEmbedding2D(nn.Module):
    """Implements 2D sinusoidal embeddings. See p.7 of Singh et al. for more details."""

    def __init__(self, d_model, max_len=100):
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
        assert w <= self.pe_x.size(0) and h <= self.pe_y.size(0), (
            f"Stored 2D positional embedding does not have enough dimensions for the current feature map. "
            f"Currently max_len={self.pe_x.size(0)}, whereas the current feature map is of shape ({w}, {h}). "
            f"Consider increasing max_len such that max_len >= T."
        )

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

    vocab_len: int
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
        vocab_len: int,
        max_seq_len: int,
        eos_tkn_idx: int,
        sos_tkn_idx: int,
        pad_tkn_idx: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
    ):
        super().__init__()
        assert d_model % 4 == 0

        self.vocab_len = vocab_len
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

        self.emb = nn.Embedding(vocab_len, d_model)
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
        self.clf = nn.Linear(d_model, vocab_len)
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
        eos_sampled = torch.zeros(B).bool()
        for t in range(self.max_seq_len):
            tgt_mask = self.subsequent_mask(len(sampled_ids)).to(memory.device)
            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # out: (B, T, d_model)
            logits = self.clf(out[:, -1, :])  # logits: (B, vocab_size)
            _, pred = torch.max(logits, -1)
            all_logits.append(logits)
            sampled_ids.append(pred)
            for i, pr in enumerate(pred):
                # Check if <EOS> is sampled for each token sequence in the batch.
                if pr == self.eos_tkn_idx:
                    eos_sampled[i] = True
            if eos_sampled.all():
                break
            tgt_ext = self.drop(
                self.pos_emb.pe[:, len(sampled_ids)]
                + self.emb(pred) * math.sqrt(self.d_model)
            ).unsqueeze(1)
            tgt = torch.cat([tgt, tgt_ext], 1)
        sampled_ids = torch.stack(sampled_ids, 1)
        all_logits = torch.stack(all_logits, 1)

        # Replace all tokens in `sampled_ids` after <EOS> with <PAD> tokens.
        eos_idxs = (sampled_ids == self.eos_tkn_idx).float().argmax(1)
        for i in range(B):
            if eos_idxs[i] != 0:  # sampled sequence contains <EOS> token
                sampled_ids[i, eos_idxs[i] + 1 :] = self.pad_tkn_idx

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

    def __init__(self, d_model: int, model_name: str, dropout: float):
        super().__init__()

        assert d_model % 4 == 0
        _models = ["resnet18", "resnet34", "resnet50"]
        err_message = f"{model_name} is not an available option: {_models}"
        assert model_name in _models, err_message

        self.d_model = d_model
        self.model_name = model_name
        self.pos_emb = PositionalEmbedding2D(d_model)
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

    def forward(self, imgs):
        x = self.encoder(imgs.unsqueeze(1))  # x: (B, d_model, w, h)
        x = self.linear(x).transpose(1, 2).transpose(2, 3)  # x: (B, w, h, d_model)
        x = self.pos_emb(x)  # x: (B, w, h, d_model)
        x = self.drop(x)  # x: (B, w, h, d_model)
        x = x.flatten(1, 2)  # x: (B, w*h, d_model)
        return x


class FullPageHTREncoderDecoder(nn.Module):
    encoder: FullPageHTREncoder
    decoder: FullPageHTRDecoder
    cer_metric: CharacterErrorRate
    wer_metric: WordErrorRate
    loss_fn: Callable
    label_encoder: LabelEncoder

    def __init__(
        self,
        label_encoder: LabelEncoder,
        max_seq_len: int = 500,
        d_model: int = 260,
        num_layers: int = 6,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        encoder_name: str = "resnet18",
        drop_enc: int = 0.1,
        drop_dec: int = 0.1,
        activ_dec: str = "gelu",
        label_smoothing: float = 0.0,
        vocab_len: Optional[int] = None,
    ):
        """
        Model used in Singh et al. (2021). The default hyperparameters are those used
        in the paper, whenever they were available.

        Args:
            label_encoder (LabelEncoder): Label encoder, which provides an
                integer encoding of token values.
            d_model (int): the number of expected features in the decoder inputs
            num_layers (int): the number of sub-decoder-layers in the decoder
            nhead (int): the number of heads in the multi-head attention models
            dim_feedforward (int): the dimension of the feedforward network model in
                the decoder
            encoder_name (str): name of the ResNet decoder to use. Choices:
                (resnet18, resnet34, resnet50)
            drop_enc (int): dropout rate used in the encoder
            drop_dec (int): dropout rate used in the decoder
            activ_dec (str): activation function of the decoder
            label_smoothing (float): label smoothing epsilon for the cross-entropy
                loss (0.0 indicates no smoothing)
            vocab_len (Optional[int]): length of the vocabulary. If passed,
                it is used rather than the length of the classes in the label encoder
        """
        super().__init__()

        # Obtain special token indices.
        self.eos_tkn_idx, self.sos_tkn_idx, self.pad_tkn_idx = label_encoder.transform(
            ["<EOS>", "<SOS>", "<PAD>"]
        )

        # Initialize encoder and decoder.
        self.encoder = FullPageHTREncoder(
            d_model, model_name=encoder_name, dropout=drop_enc
        )
        self.decoder = FullPageHTRDecoder(
            vocab_len=(vocab_len or label_encoder.n_classes),
            max_seq_len=max_seq_len,
            eos_tkn_idx=self.eos_tkn_idx,
            sos_tkn_idx=self.sos_tkn_idx,
            pad_tkn_idx=self.pad_tkn_idx,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=drop_dec,
            activation=activ_dec,
        )
        self.label_encoder = label_encoder

        # Initialize metrics and loss function.
        self.cer_metric = CharacterErrorRate(label_encoder)
        self.wer_metric = WordErrorRate(label_encoder)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad_tkn_idx, label_smoothing=label_smoothing
        )

    def forward(
        self, imgs: Tensor, targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Union[Tensor, None]]:
        """
        Run inference on the model using greedy decoding.

        Returns:
            - logits, obtained at each time step during decoding
            - sampled class indices, i.e. model predictions, obtained by applying
                  greedy decoding (argmax on logits) at each time step
            - loss value (only calculated when specifiying `targets`, otherwise
                  defaults to None)
        """
        logits, sampled_ids = self.decoder(self.encoder(imgs))
        loss = None
        if targets is not None:
            loss = self.loss_fn(
                logits[:, : targets.size(1), :].transpose(1, 2),
                targets[:, : logits.size(1)],
            )
        return logits, sampled_ids, loss

    def forward_teacher_forcing(
        self, imgs: Tensor, targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Run inference on the model using greedy decoding and teacher forcing.

        Teacher forcing implies that at each decoding time step, the ground truth
        target of the previous time step is fed as input to the model.

        Returns:
            - logits, obtained at each time step during decoding
            - loss value
        """
        memory = self.encoder(imgs)
        logits = self.decoder.decode_teacher_forcing(memory, targets)
        loss = self.loss_fn(logits.transpose(1, 2), targets)
        return logits, loss

    def calculate_metrics(self, preds: Tensor, targets: Tensor) -> Dict[str, float]:
        self.cer_metric.reset()
        self.wer_metric.reset()
        cer = self.cer_metric(preds, targets)
        wer = self.wer_metric(preds, targets)
        return {"char_error_rate": cer, "word_error_rate": wer}

    def set_num_output_classes(self, n_classes: int):
        """
        Set number of output classes of the model. This has an effect on the
        final classification layer and the token embeddings. This is
        useful for finetuning a trained model with additional output classes.
        """
        assert n_classes >= self.decoder.vocab_len, (
            "Currently, can only add classes, " "not remove them."
        )
        print(
            "Re-initializing the classification layer of the model. This is "
            "intended behavior if you initalize model training from a trained model."
        )
        # Re-initialize classification layer.
        old_vocab_len = self.decoder.vocab_len
        self.decoder.vocab_len = n_classes
        self.decoder.clf = nn.Linear(self.decoder.d_model, n_classes)

        # Add additional token embeddings for the new classes.
        # NOTE: we are assuming here that the first n embeddings from the old model
        # are still indexed in the same way, i.e. the new classes are given indices
        # starting from index n.
        new_embs = nn.Embedding(n_classes, self.decoder.d_model)
        with torch.no_grad():
            new_embs.weight[:old_vocab_len] = self.decoder.emb.weight
            self.decoder.emb = new_embs

    @staticmethod
    def full_page_htr_optimizer_params() -> Dict[str, Any]:
        """Optimizer parameters used in Singh et al., see page 9."""
        return {"optimizer_name": "Adam", "lr": 0.0002, "betas": (0.9, 0.999)}
