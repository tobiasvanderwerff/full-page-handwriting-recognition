import xml.etree.ElementTree as ET
import html
import random
from pathlib import Path
from typing import Union, Tuple, Dict, Sequence, Optional, List

import pandas as pd
import cv2 as cv
import albumentations as A
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image

from util import read_xml, find_child_by_tag, set_seed, LabelEncoder
from transforms import IAMImageTransforms


class IAMDataset(Dataset):
    MEAN = 0.8275
    STD = 0.2314
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
    parse_method: str
    only_lowercase: bool
    transforms: Optional[A.Compose]
    id_to_idx: Dict[str, int]
    _split: str
    _return_writer_id: Optional[bool]

    def __init__(
        self,
        root: Union[Path, str],
        parse_method: str,
        split: str,
        return_writer_id: bool = False,
        only_lowercase: bool = False,
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
        self.only_lowercase = only_lowercase
        self.root = Path(root)
        self.label_enc = label_enc
        self.parse_method = parse_method

        # Process the data.
        if not hasattr(self, "data"):
            if self.parse_method == "form":
                self.data = self._get_forms()
            elif self.parse_method == "word":
                self.data = self._get_words(skip_bad_segmentation=True)
            elif self.parse_method == "line":
                self.data = self._get_lines()

        # Create the label encoder.
        if self.label_enc is None:
            vocab = [self._pad_token, self._sos_token, self._eos_token]
            s = "".join(self.data["target"].tolist())
            if self.only_lowercase:
                s = s.lower()
            vocab += sorted(list(set(s)))
            self.label_enc = LabelEncoder().fit(vocab)
        if not "target_enc" in self.data.columns:
            self.data.insert(
                2,
                "target_enc",
                self.data["target"].apply(
                    lambda s: np.array(
                        self.label_enc.transform(
                            [c for c in (s.lower() if self.only_lowercase else s)]
                        )
                    )
                ),
            )

        self.transforms = self._get_transforms(split)
        self.id_to_idx = {
            Path(self.data.iloc[i]["img_path"]).stem: i for i in range(len(self))
        }

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
        return self.label_enc.classes

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
        else:  # word or line
            max_img_h = self.MAX_FORM_HEIGHT
        transforms = IAMImageTransforms(
            (max_img_h, max_img_w), self.parse_method, (IAMDataset.MEAN, IAMDataset.STD)
        )

        if split == "train":
            return transforms.train_trnsf
        elif split == "test" or split == "val":
            return transforms.test_trnsf

    def statistics(self) -> Dict[str, float]:
        assert len(self) > 0
        tmp = self.transforms
        self.transforms = None
        mean, std, cnt = 0, 0, 0
        for img, _ in self:
            mean += np.mean(img)
            std += np.var(img)
            cnt += 1
        mean /= cnt
        std = np.sqrt(std / cnt)
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


class IAMSyntheticDataGenerator(Dataset):
    """
    Data generator that creates synthetic line/form images by stitching together word
    images from the IAM dataset.
    Calling `__getitem__()` samples a newly generated synthetic image every time
    it is called.
    """

    PUNCTUATION = [",", ".", ";", ":", "'", '"', "!", "?"]

    def __init__(
        self,
        iam_root: Union[str, Path],
        label_encoder: Optional[LabelEncoder] = None,
        transforms: Optional[A.Compose] = None,
        line_width: Tuple[int, int] = (1500, 2000),
        lines_per_form: Tuple[int, int] = (1, 11),
        words_per_line: Tuple[int, int] = (4, 10),
        words_per_sequence: Tuple[int, int] = (7, 13),
        px_between_lines: Tuple[int, int] = (25, 100),
        px_between_words: int = 50,
        px_around_image: Tuple[int, int] = (100, 200),
        sample_form: bool = False,
        only_lowercase: bool = False,
        rng_seed: int = 0,
        max_height: Optional[int] = None,
    ):
        super().__init__()
        self.iam_root = iam_root
        self.label_enc = label_encoder
        self.transforms = transforms
        self.line_width = line_width
        self.lines_per_form = lines_per_form
        self.words_per_line = words_per_line
        self.words_per_sequence = words_per_sequence
        self.px_between_lines = px_between_lines
        self.px_between_words = px_between_words
        self.px_around_image = px_around_image
        self.sample_form = sample_form
        self.only_lowercase = only_lowercase
        self.rng_seed = rng_seed
        self.max_height = max_height

        self.iam_words = IAMDataset(
            iam_root,
            "word",
            "test",
            only_lowercase=only_lowercase,
        )
        if self.max_height is None:
            self.max_height = IAMDataset.MAX_FORM_HEIGHT
        if sample_form and "\n" not in self.label_encoder.classes:
            # Add the `\n` token to the label encoder (since forms can contain newlines)
            self.label_encoder.add_classes(["\n"])
        self.iam_words.transforms = None
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        # This dataset does not have a finite length since it can generate random
        # images at will, so return 1.
        return 1

    @property
    def label_encoder(self):
        if self.label_enc is not None:
            return self.label_enc
        return self.iam_words.label_enc

    def __getitem__(self, *args, **kwargs):
        """By calling this method, a newly generated synthetic image is sampled."""
        if self.sample_form:
            img, target = self.generate_form()
        else:
            img, target = self.generate_line()
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        # Encode the target sequence using the label encoder.
        target_enc = np.array(self.label_encoder.transform([c for c in target]))
        return img, target_enc

    def generate_line(self) -> Tuple[np.ndarray, str]:
        words_to_sample = self.rng.integers(*self.words_per_line)
        line_width = self.rng.integers(*self.line_width)
        return self.sample_lines(words_to_sample, line_width, sample_one_line=True)

    def generate_form(self) -> Tuple[np.ndarray, str]:
        # Randomly pick the number of words and inter-line distance in the form.
        words_to_sample = self.rng.integers(*self.lines_per_form) * 7  # 7 is handpicked
        px_between_lines = self.rng.integers(*self.px_between_lines)

        # Sample line images.
        line_width = self.rng.integers(*self.line_width)
        lines, target = self.sample_lines(words_to_sample, line_width)

        # Concatenate the lines vertically.
        form_w = max(l.shape[1] for l in lines)
        form_h = sum(l.shape[0] + px_between_lines for l in lines)
        if form_h > self.max_height:
            print(
                "Generated form height exceeds maximum height. Generating a new form."
            )
            return self.generate_form()
        form = np.ones((form_h, form_w), dtype=lines[0].dtype) * 255
        curr_h = 0
        for line_img in lines:
            h, w = line_img.shape
            form[curr_h : curr_h + h, :w] = line_img
            curr_h += h + px_between_lines

        # Add a random amount of padding around the image.
        pad_px = self.rng.integers(*self.px_around_image)
        new_h, new_w = form.shape[0] + pad_px * 2, form.shape[1] + pad_px * 2
        form = A.PadIfNeeded(
            new_h, new_w, border_mode=cv.BORDER_CONSTANT, value=255, always_apply=True
        )(image=form)["image"]

        return form, target

    def set_rng(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def sample_word_image(self) -> Tuple[np.ndarray, str]:
        idx = random.randint(0, len(self.iam_words) - 1)
        img, target = self.iam_words[idx]
        target = "".join(self.iam_words.label_enc.inverse_transform(target))
        return img, target

    def sample_word_image_sequence(
        self, words_to_sample: int
    ) -> List[Tuple[np.ndarray, str]]:
        """Sample a sequence of contiguous words."""
        assert words_to_sample >= 1
        start_idx = random.randint(0, len(self.iam_words) - 1)

        img_idxs = [start_idx]
        img_path = Path(self.iam_words.data.iloc[start_idx]["img_path"])
        _, _, line_id, word_id = img_path.stem.split("-")
        sampled_words = 1
        while sampled_words < words_to_sample:
            word_id = f"{int(word_id) + 1 :02}"
            img_name = (
                "-".join(img_path.stem.split("-")[:-2] + [line_id, word_id]) + ".png"
            )
            if not (img_path.parent / img_name).is_file():
                # Previous image was the last on its line. Go to the next line.
                line_id = f"{int(line_id) + 1 :02}"
                word_id = "00"
                img_name = (
                    "-".join(img_path.stem.split("-")[:-2] + [line_id, word_id])
                    + ".png"
                )
            if not (img_path.parent / img_name).is_file():
                # End of the document.
                return self.sample_word_image_sequence(words_to_sample)
            # Find the dataset index for the sampled word.
            ix = self.iam_words.id_to_idx.get(Path(img_name).stem)
            if ix is None:
                # If the image has segmentation=err attribute, it will
                # not be in the dataset. In this case try again.
                return self.sample_word_image_sequence(words_to_sample)
            img_idxs.append(ix)
            sampled_words += 1

        imgs, targets = zip(*[self.iam_words[idx] for idx in img_idxs])
        targets = [
            "".join(self.iam_words.label_enc.inverse_transform(t)) for t in targets
        ]
        return list(zip(imgs, targets))

    def sample_lines(
        self, words_to_sample: int, max_line_width: int, sample_one_line: bool = False
    ) -> Tuple[Union[List[np.ndarray], np.ndarray], str]:
        """
        Calls `sample_word_image_sequence` several times, using some heuristics
        to glue the sequences together.

        Returns:
            - list of line images
            - transcription for all lines combined
        """
        curr_pos, sampled_words = 0, 0
        imgs, targets, lines = [], [], []
        target_str, last_target = "", ""

        # Sample images.
        while sampled_words < words_to_sample:
            words_per_seq = self.rng.integers(*self.words_per_sequence)
            # Sample a sequence of contiguous words.
            img_tgt_seq = self.sample_word_image_sequence(words_per_seq)
            for i, (img, tgt) in enumerate(img_tgt_seq):
                # Add the sequence to the sampled words so far.
                if sampled_words >= words_to_sample:
                    break
                h, w = img.shape

                if curr_pos + w > max_line_width:
                    # Concatenate the sampled images into a line.
                    line = self.concatenate_line(imgs, targets, max_line_width)

                    if sample_one_line:
                        return line, target_str

                    lines.append(line)
                    target_str += "\n"
                    last_target = "\n"
                    curr_pos = 0
                    imgs, targets = [], []

                # Basic heuristics to avoid some strange looking sentences.
                if i == 0 and (
                    (last_target in self.PUNCTUATION and tgt in self.PUNCTUATION)
                    or (tgt in self.PUNCTUATION and sampled_words == 0)
                ):
                    continue

                if (
                    sampled_words == 0
                    or tgt in [c for c in self.PUNCTUATION if c not in ["'", '"']]
                    or last_target == "\n"
                ):
                    target_str += tgt
                else:
                    target_str += " " + tgt

                targets.append(tgt)
                imgs.append(img)

                sampled_words += 1
                last_target = tgt
                if tgt in self.PUNCTUATION:
                    # Reduce horizontal spacing for punctuation tokens.
                    curr_pos = max(0, curr_pos - self.px_between_words)
                curr_pos += w + self.px_between_words
        if imgs and targets:
            # Concatenate the remaining images into a new line.
            line = self.concatenate_line(imgs, targets, max_line_width)
            lines.append(line)
            if sample_one_line:
                return line, target_str
        return lines, target_str

    def concatenate_line(
        self, imgs: List[np.ndarray], targets: List[str], line_width: int
    ) -> np.ndarray:
        """
        Concatenate a series of (img, target) tuples into a line to create a line image.
        """
        assert len(imgs) == len(targets)

        line_height = max(im.shape[0] for im in imgs)
        line = np.ones((line_height, line_width), dtype=imgs[0].dtype) * 255

        curr_pos = 0
        prev_lower_bound = line_height
        for img, tgt in zip(imgs, targets):
            h, w = img.shape
            # Center the image in the middle of the line.
            start_h = min(max(0, int((line_height - h) / 2)), line_height - h)

            if tgt in [",", "."]:
                # If sampled a comma or dot, place them at the bottom of the line.
                start_h = min(max(0, prev_lower_bound - int(h / 2)), line_height - h)
            elif tgt in ['"', "'"]:
                # If sampled a quote, place them at the top of the line.
                start_h = 0
            if tgt in self.PUNCTUATION:
                # Reduce horizontal spacing for punctuation tokens.
                curr_pos = max(0, curr_pos - self.px_between_words)

            assert curr_pos + w <= line_width, f"{curr_pos + w} > {line_width}"
            assert start_h + h <= line_height, f"{start_h + h} > {line_height}"

            # Concatenate the word image to the line.
            line[start_h : start_h + h, curr_pos : curr_pos + w] = img

            curr_pos += w + self.px_between_words
            prev_lower_bound = start_h + h
        return line

    @staticmethod
    def get_worker_init_fn():
        def worker_init_fn(worker_id: int):
            set_seed(worker_id)
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset  # the dataset copy in this worker process
            if hasattr(dataset, "set_rng"):
                dataset.set_rng(worker_id)
            else:  # dataset is instance of `IAMDatasetSynthetic` class
                dataset.synth_dataset.set_rng(worker_id)

        return worker_init_fn


class IAMDatasetSynthetic(Dataset):
    """
    A Pytorch dataset combining the IAM dataset with the IAMSyntheticDataGenerator
    dataset.

    The distribution of real/synthetic images can be controlled by setting the
    `synth_prob` argument.
    """

    iam_dataset: IAMDataset
    synth_dataset: IAMSyntheticDataGenerator
    synth_prob: float

    def __init__(self, iam_dataset: IAMDataset, synth_prob: float = 0.3, **kwargs):
        """
        Args:
            iam_dataset (Dataset): the IAM dataset to sample from
            synth_prob (float): the probability of sampling a synthetic image when
                calling `__getitem__()`.
        """
        self.iam_dataset = iam_dataset
        self.synth_prob = synth_prob
        self.synth_dataset = IAMSyntheticDataGenerator(
            iam_root=iam_dataset.root,
            label_encoder=iam_dataset.label_enc,
            transforms=iam_dataset.transforms,
            sample_form=(True if iam_dataset.parse_method == "form" else False),
            only_lowercase=iam_dataset.only_lowercase,
            max_height=(
                (iam_dataset.data["bb_y_end"] - iam_dataset.data["bb_y_start"]).max()
                if iam_dataset.parse_method == "form"
                else None
            ),
            **kwargs,
        )

    def __getitem__(self, idx):
        iam = self.iam_dataset
        if random.random() > 1 - self.synth_prob:
            # Sample from the synthetic dataset.
            img, target = self.synth_dataset[0]
        else:
            # Index the IAM dataset.
            img, target = iam[idx]
        assert not np.any(np.isnan(img)), img
        return img, target

    def __len__(self):
        return len(self.iam_dataset)
