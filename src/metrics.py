from util import LabelEncoder

import editdistance
from torch import Tensor
from torchmetrics import Metric


class CharacterErrorRate(Metric):
    """
    Calculates Character Error Rate, calculated as Levenshtein edit distance divided
    by length of the target. Roughly speaking, this indicates the percentage or
    characters that were incorrectly predicted.
    """

    def __init__(self, label_encoder: LabelEncoder):
        super().__init__()
        self.label_encoder = label_encoder

        self.add_state("edits", default=Tensor([0]), dist_reduce_fx="sum")
        self.add_state("total_chars", default=Tensor([0]), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        Update the number of edits and ground truth characters.

        Args:
            preds (Tensor): tensor of shape (B, P), containing character predictions
            target (Tensor): tensor of shape (B, T), containing the ground truth
                character sequence
        """
        assert preds.ndim == target.ndim

        eos_tkn_idx, sos_tkn_idx = list(
            self.label_encoder.transform(["<EOS>", "<SOS>"])
        )

        if (preds[:, 0] == sos_tkn_idx).all():  # this should normally be the case
            preds = preds[:, 1:]

        eos_idxs_prd = (preds == eos_tkn_idx).float().argmax(1).tolist()
        eos_idxs_tgt = (target == eos_tkn_idx).float().argmax(1).tolist()

        for i, (p, t) in enumerate(zip(preds, target)):
            eos_idx_p, eos_idx_t = eos_idxs_prd[i], eos_idxs_tgt[i]
            p = p[:eos_idx_p] if eos_idx_p else p
            t = t[:eos_idx_t] if eos_idx_t else t
            p_str, t_str = map(tensor_to_str, (p, t))
            editd = editdistance.eval(p_str, t_str)

            self.edits += editd
            self.total_chars += t.numel()

    def compute(self) -> Tensor:
        """Compute Character Error Rate."""
        return self.edits.float() / self.total_chars


class WordErrorRate(Metric):
    """
    Calculates Word Error Rate, calculated as Levenshtein edit distance divided by
    the number of words in the target. This works the same way as Character Error
    Rate, except that we analyse at the word level, rather than the character level.
    """

    def __init__(self, label_encoder: LabelEncoder):
        super().__init__()
        self.label_encoder = label_encoder

        self.add_state("edits", default=Tensor([0]), dist_reduce_fx="sum")
        self.add_state("total_words", default=Tensor([0]), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        Update the number of edits and ground truth characters.

        Args:
            preds (Tensor): tensor of shape (B, P), containing character predictions
            target (Tensor): tensor of shape (B, T), containing the ground truth
                character sequence
        """
        assert preds.ndim == target.ndim

        eos_tkn_idx, sos_tkn_idx = self.label_encoder.transform(["<EOS>", "<SOS>"])

        if (preds[:, 0] == sos_tkn_idx).all():  # this should normally be the case
            preds = preds[:, 1:]

        eos_idxs_prd = (preds == eos_tkn_idx).float().argmax(1).tolist()
        eos_idxs_tgt = (target == eos_tkn_idx).float().argmax(1).tolist()

        for i, (p, t) in enumerate(zip(preds, target)):
            eos_idx_p, eos_idx_t = eos_idxs_prd[i], eos_idxs_tgt[i]
            p = (p[:eos_idx_p] if eos_idx_p else p).flatten().tolist()
            t = (t[:eos_idx_t] if eos_idx_t else t).flatten().tolist()
            p_words = "".join(self.label_encoder.inverse_transform(p)).split()
            t_words = "".join(self.label_encoder.inverse_transform(t)).split()
            editd = editdistance.eval(p_words, t_words)

            self.edits += editd
            self.total_words += len(t_words)

    def compute(self) -> Tensor:
        """Compute Word Error Rate."""
        return self.edits.float() / self.total_words


def tensor_to_str(t: Tensor) -> str:
    return "".join(map(str, t.flatten().tolist()))
