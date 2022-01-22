import xml.etree.ElementTree as ET
import random
import pickle
from pathlib import Path
from typing import Union, Any, List, Optional, Sequence, Dict, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from pytorch_lightning.callbacks import TQDMProgressBar


def pickle_save(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file) -> Any:
    with open(file, "rb") as f:
        return pickle.load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_xml(xml_file: Union[Path, str]) -> ET.Element:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root


def find_child_by_tag(
    xml_el: ET.Element, tag: str, value: str
) -> Union[ET.Element, None]:
    for child in xml_el:
        if child.get(tag) == value:
            return child
    return None


def matplotlib_imshow(
    img: torch.Tensor, mean: float = 0.5, std: float = 0.5, one_channel=True
):
    assert img.device.type == "cpu"
    if one_channel and img.ndim == 3:
        img = img.mean(dim=0)
    img = img * std + mean  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class LabelEncoder:
    classes: Optional[List[str]]
    idx_to_cls: Optional[Dict[int, str]]
    cls_to_idx: Optional[Dict[str, int]]
    n_classes: Optional[int]

    def __init__(self):
        self.classes = None
        self.idx_to_cls = None
        self.cls_to_idx = None
        self.n_classes = None

    def transform(self, classes: Sequence[str]) -> List[int]:
        self.check_is_fitted()
        return [self.cls_to_idx[c] for c in classes]

    def inverse_transform(self, indices: Sequence[int]) -> List[str]:
        return [self.idx_to_cls[i] for i in indices]

    def fit(self, classes: Sequence[str]):
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.idx_to_cls = dict(enumerate(classes))
        self.cls_to_idx = {cls: i for i, cls in self.idx_to_cls.items()}
        return self

    def add_classes(self, classes: List[str]):
        new_classes = self.classes + classes
        assert len(set(new_classes)) == len(
            new_classes
        ), "New labels contain duplicates"
        return self.fit(new_classes)

    def read_encoding(self, filename: Union[str, Path]):
        if Path(filename).suffix == ".pkl":
            # Label encoding saved as Sklearn LabelEncoder instance.
            return self.read_sklearn_encoding(filename)
        else:
            classes = []
            saved_str = Path(filename).read_text()
            i = 0
            while i < len(saved_str):
                # This is a bit of a roundabout way to read the saved label encoding,
                # but it is necessary in order to read special characters (like `\n`)
                # correctly.
                c = saved_str[i]
                i += 1
                while i < len(saved_str) and saved_str[i] != "\n":
                    c += saved_str[i]
                    i += 1
                classes.append(c)
                i += 1
            return self.fit(classes)

    def read_sklearn_encoding(self, filename: Union[str, Path]):
        """
        Load an encoding from a Sklearn LabelEncoder pickle. This method exists to
        maintain backwards compatability with previously saved label encoders.
        """
        label_encoder = pickle_load(filename)
        classes = list(label_encoder.classes_)

        # Check if the to-be-saved encoding is correct.
        assert (
            list(label_encoder.inverse_transform(list(range(len(classes))))) == classes
        )
        self.fit(classes)
        self.dump(Path(filename).parent)
        return self

    def dump(self, outdir: Union[str, Path]):
        """Dump the encoded labels to a txt file."""
        out = "\n".join(cls for cls in self.classes)
        (Path(outdir) / "label_encoding.txt").write_text(out)

    def check_is_fitted(self):
        if self.idx_to_cls is None or self.cls_to_idx is None:
            raise ValueError("Label encoder is not fitted yet.")


class LitProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        for k in list(items.keys()):
            if k.startswith("grad"):
                items.pop(k, None)
        items.pop("v_num", None)
        return items


def decode_prediction_and_target(
    pred: Tensor, target: Tensor, label_encoder: LabelEncoder, eos_tkn_idx: int
) -> Tuple[str, str]:
    # Find padding and <EOS> positions in predictions and targets.
    eos_idx_pred = (pred == eos_tkn_idx).float().argmax().item()
    eos_idx_tgt = (target == eos_tkn_idx).float().argmax().item()

    # Decode prediction and target.
    p, t = pred.tolist(), target.tolist()
    p = p[1:]  # skip the initial <SOS> token, which is added by default
    p = p[:eos_idx_pred] if eos_idx_pred != 0 else p
    t = t[:eos_idx_tgt] if eos_idx_tgt != 0 else t
    pred_str = "".join(label_encoder.inverse_transform(p))
    target_str = "".join(label_encoder.inverse_transform(t))
    return pred_str, target_str
