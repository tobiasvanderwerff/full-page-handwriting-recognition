import torch
from pytorch_lightning.callbacks import Callback
from sklearn.preprocessing import LabelEncoder


class ShowPredictions(Callback):
    def __init__(self, label_encoder: "LabelEncoder"):
        self.label_encoder = label_encoder

    def on_batch_end(self, trainer, pl_module: "LightningModule"):
        logits, targets = pl_module._logits, pl_module._targets
        _, preds = logits.max(-1)

        sos_idxs = (preds == pl_module.decoder.eos_tkn_idx).float().argmax(1).tolist()
        pad_idxs = (targets == pl_module.decoder.pad_tkn_idx).float().argmax(1).tolist()

        for i, (p, t) in enumerate(zip(preds.tolist(), targets.tolist())):
            max_pred_idx, max_target_idx = sos_idxs[i], pad_idxs[i]
            print("====================================")
            print("Prediction:")
            if max_pred_idx != 0:
                print("".join(self.label_encoder.inverse_transform(p)[:max_pred_idx]))
            else:
                print("".join(self.label_encoder.inverse_transform(p)))
            print("------------------------------------")
            print("Target:")
            if max_target_idx != 0:
                print("".join(self.label_encoder.inverse_transform(t)[:max_target_idx]))
            else:
                print("".join(self.label_encoder.inverse_transform(t)))
            print("====================================", end="\n\n")
