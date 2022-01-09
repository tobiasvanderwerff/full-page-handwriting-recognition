import argparse
from pathlib import Path
from typing import Tuple, Union, List

import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import pandas as pd

from lit_models import LitFullPageHTREncoderDecoder
from transforms import IAMImageTransforms
from util import LabelEncoder


IMG_SCALES_GRID_SEARCH = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


def run_model(
    model_path: Path,
    imgs: Union[torch.Tensor, List[torch.Tensor]],
    label_encoder: LabelEncoder,
    print_perplexities: bool = False,
) -> str:
    # Load model.
    model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
        str(model_path),
        label_encoder=label_encoder,
    )

    # Make prediction.
    model.eval()
    if isinstance(imgs, list):  # grid search on image resolutions
        # Use perplexity as a metric to determine the best image resolution, where we
        # define perplexity as the exponential of the cross-entropy.
        perplexities, predictions = [], []
        for im in imgs:
            with torch.inference_mode():
                logits, preds, _ = model(im.unsqueeze(0))
                pred_probs, _ = F.softmax(logits.squeeze(), -1).max(-1)
                prp = torch.exp(torch.sum(-torch.log(pred_probs)) / pred_probs.size(0))

                perplexities.append(prp.item())
                predictions.append(preds.squeeze())
        best = perplexities.index(min(perplexities))
        preds = predictions[best]
        if print_perplexities:
            print(f"Perplexities: {perplexities}")
            print(
                f"Best resolution scaling factor: {IMG_SCALES_GRID_SEARCH[best]}",
                end="\n\n",
            )
    else:  # single image
        img = imgs
        with torch.inference_mode():
            _, preds, _ = model(img.unsqueeze(0))
            preds.squeeze_()

    # Decode the prediction.
    sampled_ids = preds.numpy()[1:]
    if sampled_ids[-1] == label_encoder.transform(["<EOS>"])[0]:
        sampled_ids = sampled_ids[:-1]
    pred_str = "".join((label_encoder.inverse_transform(sampled_ids)))

    return pred_str


def prepare_data(
    img_path: Path,
    model_path,
    data_format: str = "word",
    search_resolution: bool = False,
) -> Tuple[torch.Tensor, LabelEncoder]:
    # Load the image.
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)

    assert isinstance(img, np.ndarray), (
        f"Error: image at path {img_path} is not properly loaded. "
        f"Is there something wrong with this image?"
    )

    # Load the label encoder for the trained model.
    le_dir = model_path.parent.parent
    le_path = (
        le_dir / "label_encoder.pkl"
        if (le_dir / "label_encoder.pkl").is_file()
        else le_dir / "label_encoding.txt"
    )
    label_enc = LabelEncoder().read_encoding(le_path)

    # Apply image transforms.
    imgs = []
    if search_resolution:
        for scale in IMG_SCALES_GRID_SEARCH:
            trnsf = IAMImageTransforms(img.shape, data_format, scale=scale).test_trnsf
            imgs.append(torch.from_numpy(trnsf(image=img)["image"]))
    else:
        trnsf = IAMImageTransforms(img.shape, data_format).test_trnsf
        imgs = torch.from_numpy(trnsf(image=img)["image"])

    return imgs, label_enc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a trained model.")

    # fmt: off
    parser.add_argument("--img_path", type=Path, required=True,
                        help="Path to an image for which a prediction should be made.")
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to a trained model checkpoint, which will be loaded "
                             "for inference.")
    parser.add_argument("--data_format", type=str, choices=["form", "line", "word"],
                        required=True, help=("Data format of the image. Is it a full "
                                             "page, a line, or a word?"))
    parser.add_argument("--search_resolution", action="store_true", default=False,
                        help=("Set this flag to do a small grid search to find the best "
                              "resolution for the input image, based on model "
                              "perplexity. Try this out if the model predictions are not "
                              "satisfactory."))
    args = parser.parse_args()
    # fmt: on

    img_path, model_path = args.img_path, args.model_path
    assert model_path.is_file(), f"{model_path} does not point to a file."
    assert img_path.is_file(), f"Image path {img_path} does not point to a file."

    imgs, label_encoder = prepare_data(
        img_path, model_path, args.data_format, args.search_resolution
    )
    pred_str = run_model(model_path, imgs, label_encoder, print_perplexities=False)

    print(pred_str)
