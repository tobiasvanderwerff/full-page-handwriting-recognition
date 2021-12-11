#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import html
import pickle
import random
from math import ceil
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, Dict, Sequence, Optional, List, Any

import pandas as pd
import cv2 as cv
import albumentations as A
import torch
import numpy as np
from torch import Tensor
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image

from util import (
    read_xml,
    find_child_by_tag,
    randomly_displace_and_pad,
    dpi_adjusting,
    set_seed,
)


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
    transforms: Optional[A.Compose]
    parse_method: str
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

        self._split = split
        self._return_writer_id = return_writer_id
        self.root = Path(root)
        self.label_enc = label_enc
        self.parse_method = parse_method

        if use_cache:
            self._check_for_cache()

        # Process the data.
        if not hasattr(self, "data"):
            if self.parse_method == "form":
                self.data = self._get_forms()
            elif self.parse_method == "word":
                self.data = self._get_words(skip_bad_segmentation)
            elif self.parse_method == "line":
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
        assert isinstance(img, np.ndarray), (
            f"Error: image at path {data['img_path']} is not properly loaded. "
            f"Is there something wrong with this image?"
        )
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
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
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
        _splits = ["train", "val", "test"]
        err_message = f"{split} is not a possible split: {_splits}"
        assert split in _splits, err_message
        self.transforms = self._get_transforms(split)

    def _get_transforms(self, split: str) -> A.Compose:
        max_img_w = self.MAX_FORM_WIDTH

        if self.parse_method == "form":
            max_img_h = (self.data["bb_y_end"] - self.data["bb_y_start"]).max()
            transforms = IAMImageTransforms((max_img_h, max_img_w))
        elif self.parse_method == "word" or self.parse_method == "line":
            max_img_h = self.MAX_FORM_HEIGHT
            transforms = IAMImageTransforms((max_img_h, max_img_w), "word")

        if split == "train":
            return transforms.train_trnsf
        elif split == "test" or split == "val":
            return transforms.test_trnsf

    def _check_for_cache(self, cache_path="cache/"):
        """Check for cached files and load them if available."""
        df_path = Path(cache_path) / f"IAM-{self.parse_method}-df.pkl"
        le_path = Path(cache_path) / f"IAM-{self.parse_method}-labelenc.pkl"
        if df_path.is_file():
            self.data = pd.read_pickle(df_path)
        if le_path.is_file():
            self.label_enc = pd.read_pickle(le_path)

    def _cache_data(self, cache_path="cache/") -> None:
        """Cache the dataframe containing the data and the label encoder."""
        cache_dir = Path(cache_path)
        cache_dir.mkdir(exist_ok=True)
        df_path = cache_dir / f"IAM-{self.parse_method}-df.pkl"
        le_path = cache_dir / f"IAM-{self.parse_method}-labelenc.pkl"
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


@dataclass
class TemporalStack:
    """A stack where each item on the stack has an associated age."""

    @dataclass
    class StackItem:
        value: Any
        age: int = 0

    items: List[StackItem] = field(default_factory=list)

    def time_step(self):
        # Increment age by 1 for each item on the stack.
        for item in self.items:
            item.age += 1

    def add_item(self, x: Any):
        self.items.append(self.StackItem(x))

    def is_empty(self):
        return len(self) == 0

    def pop(self):
        return self.items.pop().value

    def __len__(self):
        return len(self.items)


class SyntheticDataGenerator(Dataset):
    PUNCTUATION = [",", ".", ";", ":", "'", '"', "!", "?"]

    def __init__(
        self,
        iam_root: Union[str, Path],
        label_encoder: Optional[LabelEncoder] = None,
        transforms: Optional[A.Compose] = None,
        words_per_line: Tuple[int, int] = (5, 13),
        lines_per_form: Tuple[int, int] = (3, 9),
        px_between_lines: Tuple[int, int] = (50, 100),
        px_between_words: int = 50,
        sample_form: bool = False,
        rng_seed: int = 0,
    ):
        super().__init__()
        self.iam_root = iam_root
        self.label_enc = label_encoder
        self.transforms = transforms
        self.words_per_line = words_per_line
        self.lines_per_form = lines_per_form
        self.px_between_lines = px_between_lines
        self.px_between_words = px_between_words
        self.sample_form = sample_form
        self.rng_seed = rng_seed

        self.images = IAMDataset(
            iam_root,
            "word",
            "test",
            use_cache=False,
            skip_bad_segmentation=True,
        )
        self.images.transforms = None

        self.rng = np.random.default_rng(rng_seed)
        self.background_smoothing_transform = A.RandomBrightnessContrast(
            always_apply=True,
            brightness_limit=(0.10, 0.15),
            contrast_limit=(0.10, 0.15),
        )

    def __getitem__(self, idx):
        # Note that idx is not used, rather a new image is generated every time this
        # method is called.
        if self.sample_form:
            img, target = self.generate_form()
        else:
            img, target = self.generate_line()
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        target_enc = self.label_encoder.transform([c for c in target.lower()])
        return img, target_enc

    def __len__(self):
        # This dataset does not have a finite length since it can generate random
        # images at will, so simply return 1.
        return 1

    @property
    def label_encoder(self):
        if self.label_enc is not None:
            return self.label_enc
        return self.images.label_enc

    @staticmethod
    def get_worker_init_fn():
        def worker_init_fn(worker_id: int):
            set_seed(worker_id)
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset  # the dataset copy in this worker process
            dataset.set_rng(worker_id)

        return worker_init_fn

    def set_rng(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def sample_image(self) -> Tuple[np.ndarray, str]:
        idx = random.randint(0, len(self.images))
        img, target = self.images[idx]
        target = "".join(self.images.label_enc.inverse_transform(target))
        return img, target

    def generate_line(self, apply_smoothing=True) -> Tuple[np.ndarray, str]:
        curr_pos, n_sampled_words = 0, 0
        imgs, targets = [], []
        target_str, last_target = "", ""
        last_target_popped = False
        img_stack = TemporalStack()

        # Determine the amount of words in the line by sampling from a discrete uniform
        # distribution.
        n_words_to_sample = self.rng.integers(*self.words_per_line)

        # TODO: right now words are sampled randomly. Instead, use a language model to
        # guide the sampled words. Temperature of the LM should be set to encourage
        # variation, since otherwise some words may never be sampled.

        # Sample images.
        while n_sampled_words < n_words_to_sample:
            # Probability of sampling from the stack is determined by an exponential
            # function of the age of the youngest item on the stack.
            min_age = 0 if img_stack.is_empty() else img_stack.items[-1].age
            pop_the_stack = (not img_stack.is_empty()) and self.rng.binomial(
                1, 1 - (0.5 ** (min_age - 1))
            )
            if pop_the_stack:
                img, tgt = img_stack.pop()
            else:  # sample a random image
                img, tgt = self.sample_image()
            h, w = img.shape

            # A basic heuristic to avoid strange looking sentences.
            if (
                last_target in self.PUNCTUATION and tgt in self.PUNCTUATION
            ) or last_target == tgt:
                continue

            if tgt in ['"', "'"] and not pop_the_stack:
                # When sampling quotation symbols, close them at a random point,
                # by adding them to a stack that will be popped later.
                img_stack.add_item((img, tgt))

            if (
                pop_the_stack
                or tgt in [c for c in self.PUNCTUATION if c not in ["'", '"']]
                or (last_target in ['"', "'"] and not last_target_popped)
                or n_sampled_words == 0
            ):
                # Surrounding quotes should not have spaces on at least one side,
                # so they should be "glued" to the next or previous target word.
                target_str += tgt
            else:
                target_str += " " + tgt
            targets.append(tgt)
            imgs.append(img)

            n_sampled_words += 1
            curr_pos += w
            if tgt not in self.PUNCTUATION:
                curr_pos += self.px_between_words
            last_target = tgt
            last_target_popped = True if pop_the_stack else False
            img_stack.time_step()

        # Append the remaining images from the stack to the line.
        while not img_stack.is_empty():
            img, tgt = img_stack.pop()
            imgs.append(img)
            targets.append(tgt)
            target_str += tgt
            curr_pos += img.shape[1]

        # Concatenate the images into a line.
        line_h = max(im.shape[0] for im in imgs)
        line_w = max(curr_pos, sum(im.shape[1] + self.px_between_words for im in imgs))
        line = np.ones((line_h, line_w), dtype=imgs[0].dtype) * 255
        curr_pos = 0
        prev_lower_bound = line_h
        assert len(imgs) == len(targets)
        for img, tgt in zip(imgs, targets):
            h, w = img.shape
            # Center the image in the middle of the line.
            start_h = min(max(0, int((line_h - h) / 2)), line_h - h)

            if tgt in [",", "."]:
                # If sampled a comma or dot, place them at the bottom of the line.
                start_h = min(max(0, prev_lower_bound - int(h / 2)), line_h - h)
            elif tgt in ['"', "'"]:
                # If sampled a quote, place them at the top of the line.
                start_h = 0
            if tgt in self.PUNCTUATION:
                # Reduce horizontal spacing for punctuation tokens.
                curr_pos = max(0, curr_pos - self.px_between_words)

            assert curr_pos + w <= line_w, f"{curr_pos + w} > {line_w}"
            assert start_h + h <= line_h, f"{start_h + h} > {line_h}"

            # Concatenate the word image to the line.
            line[start_h : start_h + h, curr_pos : curr_pos + w] = img

            curr_pos += w + self.px_between_words
            prev_lower_bound = start_h + h

        # Apply random brightness and contrast adjustment in an attempt to smooth out
        # the background of the image.
        if apply_smoothing:
            line = self.background_smoothing_transform(image=line)["image"]

        return line, target_str

    def generate_form(self) -> Tuple[np.ndarray, str]:
        target = ""
        lines = []
        n_lines_to_sample = self.rng.integers(*self.lines_per_form)
        px_between_lines = self.rng.integers(*self.px_between_lines)

        # TODO: make the numbers of words per line (roughly) uniform.

        # Sample line images.
        for i in range(n_lines_to_sample):
            line_img, line_target = self.generate_line(apply_smoothing=False)
            lines.append(line_img)
            target += line_target + "\n"
        form_w = max(l.shape[1] for l in lines)
        form_h = sum(l.shape[0] + px_between_lines for l in lines)
        form = np.ones((form_h, form_w), dtype=lines[0].dtype) * 255

        # Concatenate the lines vertically.
        curr_h = 0
        for line_img in lines:
            h, w = line_img.shape
            form[curr_h : curr_h + h, :w] = line_img
            curr_h += h + px_between_lines

        form = self.background_smoothing_transform(image=form)["image"]

        return form, target


class IAMDatasetSynthetic(Dataset):
    """
    A Pytorch dataset combining the IAM dataset with the SyntheticDataGenerator
    dataset.
    """

    iam_dataset: IAMDataset
    synth_dataset: SyntheticDataGenerator
    synth_prob: float

    def __init__(self, iam_dataset: IAMDataset, synth_prob: float = 0.3):
        """
        Args:
            iam_dataset (Dataset): the IAM dataset to sample from
            synth_prob (float): the probability of sampling a synthetic image when
                calling `__getitem__()`.
        """
        self.iam_dataset = iam_dataset
        self.synth_prob = synth_prob
        self.synth_dataset = SyntheticDataGenerator(
            iam_root=iam_dataset.root,
            label_encoder=iam_dataset.label_enc,
            transforms=iam_dataset.transforms,
            sample_form=(True if iam_dataset.parse_method == "form" else False),
        )

    def __getitem__(self, idx):
        iam = self.iam_dataset
        if random.random() > 1 - self.synth_prob:
            # Sample from the synthetic dataset.
            img, target = self.synth_dataset[0]
        else:
            # Index the IAM dataset.
            img, target = iam[idx]
        return img, target

    def __len__(self):
        return len(self.iam_dataset)
