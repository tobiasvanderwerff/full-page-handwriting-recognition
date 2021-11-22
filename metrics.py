import editdistance
from torch import Tensor
from torchmetrics import Metric, WER


class CharacterErrorRate(Metric):
    """
    Calculates Character Error Rate, i.e. Levenshtein edit distance divided by length of
    ground truth. Roughly speaking, this indicates the percentage or characters that
    were incorrectly predicted.
    """

    def __init__(self):
        super().__init__()
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

        def to_str(t: Tensor) -> str:
            return "".join(map(str, t.flatten().tolist()))

        pred_str, tgt_str = map(to_str, (preds, target))
        editd = editdistance.eval(pred_str, tgt_str)

        self.edits += editd
        self.total_chars += target.numel()

    def compute(self) -> Tensor:
        """Compute Character Error Rate."""
        return self.edits.float() / self.total_chars


# class WordErrorRate(WER):
#     def __init__(self):
#         super().__init__()

#     def to_str(self, preds: Tensor, target: Tensor):
#         def _to_str(t: Tensor) -> str:
#             return ''.join(map(str, t.flatten().tolist()))

#         # device = self.preds.device
#         pred_str, tgt_str = map(to_str, (preds, target))
#         return pred_str, tgt_str
