import logging
import base64
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import albumentations as A
import cv2 as cv
from ts.torch_handler.vision_handler import VisionHandler
from torch.profiler import ProfilerActivity

from fphtr.lit_models import LitFullPageHTREncoderDecoder
from fphtr.transforms import IAMImageTransforms


logger = logging.getLogger(__name__)


class ImageTextTranscription(VisionHandler):
    """
    Handler class. This handler extends class ImageClassifier, a default handler. This
    handler takes an image and returns the transcription of the text in that image.

    Here method postprocess() has been overridden while others are reused from parent
    class.
    """

    IMG_SCALES_GRID_SEARCH = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    def __init__(self):
        super().__init__()
        self.label_encoder = None
        self.profiler_args = {
            "activities": [ProfilerActivity.CPU],
            "record_shapes": True,
        }

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = Path(properties.get("model_dir"))

        assert "serializedFile" in self.manifest["model"]
        serialized_file = self.manifest["model"]["serializedFile"]
        model_ckpt_path = model_dir / serialized_file

        logger.debug("Loading eager model")

        # # model def file
        # model_file = Path(self.manifest["model"].get("modelFile", ""))
        # model_def_path = model_dir / model_file
        # if not model_def_path.is_file():
        #     raise RuntimeError("Missing the model.py file")

        # Load the label encoder for the trained model.
        self.label_encoder = pd.read_pickle(model_dir / "label_encoder.pkl")

        # Load model.
        self.model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
            str(model_ckpt_path),
            label_encoder=self.label_encoder,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.debug(f"Model file {model_ckpt_path} loaded successfully")

        self.initialized = True

    def preprocess(self, data) -> torch.Tensor:
        """
        Preprocess function to convert the request input to a tensor (Torchserve
        supported format). The user needs to override to customize the pre-processing.

        Args:
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """

        # logger.info(f"length of `data`: {len(data)}")
        # logger.info(f"type of `data` element: {type(data[0])}")
        # logger.info(f"`data[0]`: {data[0]}")

        assert len(data) == 1

        for row in data:
            img = row.get("data") or row.get("body")
            if isinstance(img, str):
                # if the image is a string of bytesarray.
                img = base64.b64decode(img)

            # If the image is sent as bytesarray
            if isinstance(img, (bytearray, bytes)):
                img = np.frombuffer(img, dtype=np.uint8)
                img = cv.imdecode(img, cv.IMREAD_GRAYSCALE)
                # img = img.open(io.BytesIO(img))
                # img = self.image_processing(img)
            else:
                # if the image is a list
                img = torch.FloatTensor(img)

        # img = VisionHandler.preprocess(self, b64_data).squeeze()

        # Find the best image scale.
        imgs = []
        h, w = img.shape
        for scale in self.IMG_SCALES_GRID_SEARCH:
            # Apply image transforms.
            trnsf = IAMImageTransforms((0, 0), "line", scale=scale).test_trnsf
            imgs.append(trnsf(image=img)["image"])

        # Pad the images to fit in one batch.
        # TODO: check if this padding is potentially harmful to performance.
        img_sizes = [im.shape for im in imgs]
        hs, ws = zip(*img_sizes)
        pad_fn = A.PadIfNeeded(
            max(hs), max(ws), border_mode=cv.BORDER_CONSTANT, value=0
        )
        imgs = [pad_fn(image=im)["image"] for im in imgs]

        res = torch.from_numpy(np.stack(imgs))
        logger.info(f"Shape for inference: {res.shape}")
        return res

    def inference(
        self, imgs: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The Inference Function is used to make a prediction call on the given input request.

        Args:
            imgs (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        imgs = imgs.to(self.device)
        with torch.inference_mode():
            logits, preds, _ = self.model(imgs)
        return logits, preds

    def postprocess(self, output: Tuple[torch.Tensor, torch.Tensor]) -> List[str]:
        """The post process decodes the predicted output response to a string.

        Args:
            output: tuple of the form (logits, preds).
        Returns:
            list: A list of dictionaries with predictions and explanations is returned
        """
        logits, preds = output

        eos_tkn_idx = self.label_encoder.transform(["<EOS>"])[0]

        # Use perplexity as a metric to determine the best image resolution, where we
        # define perplexity as the exponential of the cross-entropy.
        perplexities, predictions = [], []
        probs, _ = F.softmax(logits, -1).max(-1)
        for prd, prb in zip(preds, probs):
            eos_idx = (prd == eos_tkn_idx).float().argmax().item()
            if eos_idx != 0:
                prd, prb = prd[:eos_idx], prb[:eos_idx]
            prp = torch.exp(torch.sum(-torch.log(prb)) / prb.size(0)).item()

            perplexities.append(prp)
            predictions.append(prd)

        best = perplexities.index(min(perplexities))
        preds = predictions[best]

        logger.info(f"Perplexities: {perplexities}")
        logger.info(
            f"Best resolution scaling factor: {self.IMG_SCALES_GRID_SEARCH[best]}",
        )

        # Decode the prediction to a string.
        sampled_ids = preds.numpy()[1:]
        if sampled_ids[-1] == eos_tkn_idx:
            sampled_ids = sampled_ids[:-1]
        pred_str = "".join((self.label_encoder.inverse_transform(sampled_ids)))

        return [pred_str]
