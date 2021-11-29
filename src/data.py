#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import html
import pickle
import random
from math import ceil
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, Dict, Sequence, Optional

import pandas as pd
import cv2 as cv
import albumentations as A
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from util import read_xml, find_child_by_tag, randomly_displace_and_pad, dpi_adjusting


@dataclass
class IAMImageTransforms:
    """Image transforms for the IAM dataset.

    All images are padded to the same size. Images are randomly displaced
    before padding during training, and centered during validation and
    testing.
    """

    max_img_size: Tuple[int, int]  # (h, w)
    parse_method: Optional[str] = ""
    scale: int = (
        0.5  # assuming A4 paper, this gives ~140 DPI (see Singh et al. p. 8, section 4)
    )
    random_scale_limit: float = 0.1
    random_rotate_limit: int = 10
    normalize_params: Tuple[float, float] = (
        0.5,
        0.5,
    )  # TODO: find proper normalization params
    train_trnsf: A.Compose = field(init=False)
    test_trnsf: A.Compose = field(init=False)

    def __post_init__(self):
        scale, random_scale_limit, random_rotate_limit, normalize_params = (
            self.scale,
            self.random_scale_limit,
            self.random_rotate_limit,
            self.normalize_params,
        )
        max_img_h, max_img_w = self.max_img_size

        max_scale = scale + scale * random_scale_limit
        padded_h, padded_w = ceil(max_scale * max_img_h), ceil(max_scale * max_img_w)

        if self.parse_method == "word":
            self.train_trnsf = A.Compose(
                [
                    A.Lambda(partial(dpi_adjusting, scale=scale)),
                    A.SafeRotate(
                        limit=random_rotate_limit,
                        border_mode=cv.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.RandomBrightnessContrast(),
                    A.GaussNoise(),
                    A.Normalize(*normalize_params),
                ]
            )
            self.test_trnsf = A.Compose(
                [
                    A.Lambda(partial(dpi_adjusting, scale=scale)),
                    A.Normalize(*normalize_params),
                ]
            )
        elif self.parse_method == "line":
            self.train_trnsf = A.Compose(
                [
                    A.Lambda(partial(dpi_adjusting, scale=scale)),
                    A.RandomScale(scale_limit=random_scale_limit, p=0.5),
                    A.SafeRotate(
                        limit=random_rotate_limit,
                        border_mode=cv.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.RandomBrightnessContrast(),
                    A.GaussNoise(),
                    A.Normalize(*normalize_params),
                ]
            )
            self.test_trnsf = A.Compose(
                [
                    A.Lambda(partial(dpi_adjusting, scale=scale)),
                    A.Normalize(*normalize_params),
                ]
            )
        else:  # forms by default
            self.train_trnsf = A.Compose(
                [
                    A.Lambda(partial(dpi_adjusting, scale=scale)),
                    A.RandomScale(scale_limit=random_scale_limit, p=0.5),
                    # SafeRotate is preferred over Rotate because it does not cut off text
                    # when it extends out of the frame after rotation.
                    A.SafeRotate(
                        limit=random_scale_limit,
                        border_mode=cv.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.RandomBrightnessContrast(),
                    A.Perspective(scale=(0.03, 0.05)),
                    A.GaussNoise(),
                    A.Lambda(
                        image=partial(
                            randomly_displace_and_pad, padded_size=(padded_h, padded_w)
                        )
                    ),
                    A.Normalize(*normalize_params),
                ]
            )
            self.test_trnsf = A.Compose(
                [
                    A.Lambda(partial(dpi_adjusting, scale=scale)),
                    A.PadIfNeeded(
                        padded_h, padded_w, border_mode=cv.BORDER_CONSTANT, value=0
                    ),
                    A.Normalize(*normalize_params),
                ]
            )


class IAMDataset(Dataset):
    MAX_FORM_HEIGHT = 3542
    MAX_FORM_WIDTH = 2479
    MAX_SEQ_LENS = {
        "word": 55,
        "line": 90,
        "form": 700,
    }  # based on the maximum seq lengths found in the dataset

    _pad_token = "<PAD>"
    _sos_token = "<SOS>"
    _eos_token = "<EOS>"

    root: Path
    data: pd.DataFrame
    label_enc: LabelEncoder
    transforms: A.Compose
    _parse_method: str
    _split: str

    def __init__(
        self,
        root: Union[Path, str],
        parse_method: str,
        split: str,
        use_cache: bool = False,
        skip_bad_segmentation: bool = False,
        return_writer_id: bool = False,
        label_enc: Optional[LabelEncoder] = None,
    ):
        super().__init__()
        _parse_methods = ["form", "line", "word"]
        err_message = (
            f"{parse_method} is not a possible parsing method: {_parse_methods}"
        )
        assert parse_method in _parse_methods, err_message

        _splits = ["train", "test"]
        err_message = f"{split} is not a possible split: {_splits}"
        assert split in _splits, err_message

        self._parse_method = parse_method
        self._split = split
        self._return_writer_id = return_writer_id
        self.root = Path(root)
        self.label_enc = label_enc

        if use_cache:
            self._check_for_cache()

        # Process the data.
        if not hasattr(self, "data"):
            if self._parse_method == "form":
                self.data = self._get_forms()
            elif self._parse_method == "word":
                self.data = self._get_words(skip_bad_segmentation)
            elif self._parse_method == "line":
                self.data = self._get_lines(skip_bad_segmentation)

        # Create the label encoder. We convert all ASCII characters to lower case.
        if self.label_enc is None:
            vocab = [self._pad_token, self._sos_token, self._eos_token]
            vocab += sorted(list(set(("".join(self.data["target"].tolist()).lower()))))
            self.label_enc = LabelEncoder().fit(vocab)
        if not "target_enc" in self.data.columns:
            self.data.insert(
                2,
                "target_enc",
                self.data["target"].apply(
                    lambda s: self.label_enc.transform([c for c in s.lower()])
                ),
            )

        self.transforms = self._get_transforms(split)

        if use_cache:
            self._cache_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        img = cv.imread(data["img_path"], cv.IMREAD_GRAYSCALE)
        if all(col in data.keys() for col in ["bb_y_start", "bb_y_end"]):
            # Crop the image vertically.
            img = img[data["bb_y_start"] : data["bb_y_end"], :]
        if not isinstance(img, np.ndarray):
            print(type(img), data["img_path"])
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        if self._return_writer_id:
            return img, data["writer_id"], data["target_enc"]
        return img, data["target_enc"]

    @property
    def vocab(self):
        return self.label_enc.classes_.tolist()

    @staticmethod
    def collate_fn(
        batch: Sequence[Tuple[np.ndarray, np.ndarray]],
        pad_val: int,
        eos_tkn_idx: int,
        dataset_returns_writer_id: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dataset_returns_writer_id:
            imgs, writer_ids, targets = zip(*batch)
        else:
            imgs, targets = zip(*batch)

        img_sizes = [im.shape for im in imgs]
        if (
            not len(set(img_sizes)) == 1
        ):  # images are of varying sizes, so pad them to the maximum size in the batch
            hs, ws = zip(*img_sizes)
            pad_fn = A.PadIfNeeded(
                max(hs), max(ws), border_mode=cv.BORDER_CONSTANT, value=0
            )
            imgs = [pad_fn(image=im)["image"] for im in imgs]
        imgs = np.stack(imgs, axis=0)

        seq_lengths = [t.shape[0] for t in targets]
        targets_padded = np.full((len(targets), max(seq_lengths) + 1), pad_val)
        for i, t in enumerate(targets):
            targets_padded[i, : seq_lengths[i]] = t
            targets_padded[i, seq_lengths[i]] = eos_tkn_idx

        imgs, targets_padded = torch.tensor(imgs), torch.tensor(targets_padded)
        if dataset_returns_writer_id:
            return imgs, targets_padded, torch.tensor(writer_ids)
        return imgs, targets_padded

    def set_transforms_for_split(self, split: str):
        _splits = ["train", "eval", "test"]
        err_message = f"{split} is not a possible split: {_splits}"
        assert split in _splits, err_message
        self.transforms = self._get_transforms(split)

    def _get_transforms(self, split: str) -> A.Compose:
        max_img_w = self.MAX_FORM_WIDTH

        if self._parse_method == "form":
            max_img_h = (self.data["bb_y_end"] - self.data["bb_y_start"]).max()
            transforms = IAMImageTransforms((max_img_h, max_img_w))
        elif self._parse_method == "word" or self._parse_method == "line":
            max_img_h = self.MAX_FORM_HEIGHT
            transforms = IAMImageTransforms((max_img_h, max_img_w), "word")

        if split == "train":
            return transforms.train_trnsf
        elif split == "test" or split == "eval":
            return transforms.test_trnsf

    def _check_for_cache(self, cache_path="cache/"):
        """Check for cached files and load them if available."""
        df_path = Path(cache_path) / f"IAM-{self._parse_method}-df.pkl"
        le_path = Path(cache_path) / f"IAM-{self._parse_method}-labelenc.pkl"
        if df_path.is_file():
            self.data = pd.read_pickle(df_path)
        if le_path.is_file():
            self.label_enc = pd.read_pickle(le_path)

    def _cache_data(self, cache_path="cache/") -> None:
        """Cache the dataframe containing the data and the label encoder."""
        cache_dir = Path(cache_path)
        cache_dir.mkdir(exist_ok=True)
        df_path = cache_dir / f"IAM-{self._parse_method}-df.pkl"
        le_path = cache_dir / f"IAM-{self._parse_method}-labelenc.pkl"
        if not df_path.is_file():
            self.data.to_pickle(df_path)
        if not le_path.is_file():
            with open(le_path, "wb") as f:
                pickle.dump(self.label_enc, f)

    def statistics(self) -> Dict[str, float]:
        tmp = self.transforms
        self.transforms = None
        imgs = torch.cat([img for img, _ in self])
        mean = torch.mean(imgs).numpy()
        std = torch.std(imgs).numpy()
        self.transforms = tmp
        return {"mean": mean, "std": std}

    def _get_forms(self) -> pd.DataFrame:
        """Read all form images from the IAM dataset.

        Returns:
            pd.DataFrame
                A pandas dataframe containing the image path, image id, target, vertical
                upper bound, vertical lower bound, and target length.
        """
        data = {
            "img_path": [],
            "img_id": [],
            "target": [],
            "bb_y_start": [],
            "bb_y_end": [],
            "target_len": [],
        }
        for form_dir in ["formsA-D", "formsE-H", "formsI-Z"]:
            dr = self.root / form_dir
            for img_path in dr.iterdir():
                doc_id = img_path.stem
                xml_root = read_xml(self.root / "xml" / (doc_id + ".xml"))

                # Based on some empiricial evaluation, the 'asy' and 'dsy'
                # attributes of a line xml tag seem to correspond to its upper and
                # lower bound, respectively. We add padding of 150 pixels.
                bb_y_start = int(xml_root[1][0].get("asy")) - 150
                bb_y_end = int(xml_root[1][-1].get("dsy")) + 150

                form_text = []
                for line in xml_root.iter("line"):
                    form_text.append(html.unescape(line.get("text", "")))

                img_w, img_h = Image.open(str(img_path)).size
                data["img_path"].append(str(img_path))
                data["img_id"].append(doc_id)
                data["target"].append("\n".join(form_text))
                data["bb_y_start"].append(bb_y_start)
                data["bb_y_end"].append(bb_y_end)
                data["target_len"].append(len("\n".join(form_text)))
        return pd.DataFrame(data).sort_values(
            "target_len"
        )  # by default, sort by target length

    def _get_lines(self, skip_bad_segmentation: bool = False) -> pd.DataFrame:
        """Read all line images from the IAM dataset.

        Args:
            skip_bad_segmentation (bool): skip lines that have the
                segmentation='err' xml attribute
        Returns:
            List of 2-tuples, where each tuple contains the path to a line image
            along with its ground truth text.
        """
        data = {"img_path": [], "img_id": [], "target": []}
        root = self.root / "lines"
        for d1 in root.iterdir():
            for d2 in d1.iterdir():
                doc_id = d2.name
                xml_root = read_xml(self.root / "xml" / (doc_id + ".xml"))
                for img_path in d2.iterdir():
                    target = self._find_line(
                        xml_root, img_path.stem, skip_bad_segmentation
                    )
                    if target is not None:
                        data["img_path"].append(str(img_path.resolve()))
                        data["img_id"].append(doc_id)
                        data["target"].append(target)
        return pd.DataFrame(data)

    def _get_words(self, skip_bad_segmentation: bool = False) -> pd.DataFrame:
        """Read all word images from the IAM dataset.

        Args:
            skip_bad_segmentation (bool): skip lines that have the
                segmentation='err' xml attribute
        Returns:
            List of 2-tuples, where each tuple contains the path to a word image
            along with its ground truth text.
        """
        data = {"img_path": [], "img_id": [], "writer_id": [], "target": []}
        root = self.root / "words"
        for d1 in root.iterdir():
            for d2 in d1.iterdir():
                doc_id = d2.name
                xml_root = read_xml(self.root / "xml" / (doc_id + ".xml"))
                writer_id = int(xml_root.get("writer-id"))
                for img_path in d2.iterdir():
                    target = self._find_word(
                        xml_root, img_path.stem, skip_bad_segmentation
                    )
                    if target is not None:
                        data["img_path"].append(str(img_path.resolve()))
                        data["img_id"].append(doc_id)
                        data["writer_id"].append(writer_id)
                        data["target"].append(target)
        return pd.DataFrame(data)

    def _find_line(
        self,
        xml_root: ET.Element,
        line_id: str,
        skip_bad_segmentation: bool = False,
    ) -> Union[str, None]:
        line = find_child_by_tag(xml_root[1].findall("line"), "id", line_id)
        if line is not None and not (
            skip_bad_segmentation and line.get("segmentation") == "err"
        ):
            return html.unescape(line.get("text"))
        return None

    def _find_word(
        self,
        xml_root: ET.Element,
        word_id: str,
        skip_bad_segmentation: bool = False,
    ) -> Union[str, None]:
        line_id = "-".join(word_id.split("-")[:-1])
        line = find_child_by_tag(xml_root[1].findall("line"), "id", line_id)
        if line is not None and not (
            skip_bad_segmentation and line.get("segmentation") == "err"
        ):
            word = find_child_by_tag(line.findall("word"), "id", word_id)
            if word is not None:
                return html.unescape(word.get("text"))
        return None


class SyntheticDataGenerator:
    # TODO: make a synthetic data generator. Some considerations:
    # 1. At what level to concatentate (e.g. word, line, sentence). What effect will
    #    this choice have on the language model that is trained?
    # 2. What heuristics to apply, e.g. minimum line length to avoid lines that are
    #    only one or a few words.

    def __init__(
        self, iam_root: str, max_line_width: int = 500, space_between_words: int = 20
    ):
        self.max_line_width = max_line_width
        self.space_between_words = space_between_words
        # self.images = IAMDataset(
        #     iam_root,
        #     "word",
        #     "test",
        #     use_cache=False,
        #     skip_bad_segmentation=True,
        # )

    def sample_image(self) -> Tuple[np.ndarray, str]:
        idx = random.randint(0, len(self.images))
        img, target = self.images[idx]
        target = "".join(self.images.label_enc.inverse_transform(target))
        return img, target

    def generate_line(self):
        curr_pos = 0
        imgs, targets, img_stack = [], [], []

        # Sample images.
        while curr_pos < self.max_line_width:
            # With 25% probability, pop an image from the stack (a closing quote).
            if len(img_stack) != 0 and (not random.randint(0, 3)):
                img, tgt = img_stack.pop()
            else:  # sample a random image
                img, tgt = self.sample_image()
            h, w = img.shape

            if curr_pos + w > self.max_line_width:
                if img_stack != []:
                    pass
                    # for img, tgt in img_stack:
                    # TODO
                    #
                    # Empty the stack
                break

            # When sampling quotes, close them at a random point.
            if tgt in ['"', "'"]:
                img_stack.append((img, tgt))

            imgs.append(img)
            targets.append(tgt)
            curr_pos += w + self.space_between_words

        # Concatenate the images into a line.
        max_h = max(im.shape[0] for im in imgs)
        line = np.ones((max_h, self.max_line_width))
        curr_pos = 0
        for img, tgt in zip(imgs, targets):
            h, w = img.shape
            start_h = int((max_h - h) / 2)

            # If sampled a comma or dot, place them at the bottom of the line
            if tgt in [",", "."]:
                start_h = max_h - h
            # If sampled a quote, place them at the top of the line
            if tgt in ['"', "'"]:
                start_h = 0

            # Concatenate the sampled word image to the line.
            line[start_h : start_h + h, curr_pos : curr_pos + w] = img

            curr_pos += w + self.space_between_words
        return line, " ".join(targets)
