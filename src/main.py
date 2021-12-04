#!/usr/bin/env python3

import argparse
import pickle
import random
from copy import copy
from pathlib import Path
from functools import partial

from models import *
from lit_models import LitFullPageHTREncoderDecoder
from lit_callbacks import LogModelPredictions
from data import IAMDataset

import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin


LOGGING_DIR = "lightning_logs/"
LOGMODELPREDICTIONS_TO_SAMPLE = 8


def main(args):

    seed_everything(args.seed)

    log_dir_root = Path(__file__).parent.parent.resolve()
    tb_logger = pl_loggers.TensorBoardLogger(log_dir_root / LOGGING_DIR, name="")

    label_enc = None
    if args.validate:
        # Load the label encoder for the trained model.
        le_path = Path(args.validate).parent.parent / "label_encoder.pkl"
        assert le_path.is_file(), (
            f"Label encoder file not found at {le_path}. "
            f"Make sure 'label_encoder.pkl' exists in the lightning_logs directory."
        )
        label_enc = pd.read_pickle(le_path)

    ds = IAMDataset(
        args.data_dir, args.data_format, "train", use_cache=False, label_enc=label_enc
    )

    if not args.validate:
        # Save the label encoder.
        save_dir = Path(tb_logger.log_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        le_path = save_dir / "label_encoder.pkl"
        if not le_path.is_file():
            with open(le_path, "wb") as f:
                pickle.dump(ds.label_enc, f)

    eos_tkn_idx, sos_tkn_idx, pad_tkn_idx = ds.label_enc.transform(
        [ds._eos_token, ds._sos_token, ds._pad_token]
    ).tolist()
    collate_fn = partial(
        IAMDataset.collate_fn, pad_val=pad_tkn_idx, eos_tkn_idx=eos_tkn_idx
    )

    # Split the dataset into train/eval/(test).
    if args.use_aachen_splits:
        # Use the Aachen splits for the IAM dataset. It should be noted that these
        # splits do not encompass the complete IAM dataset.
        aachen_path = Path(__file__).parent.parent / "aachen_splits"
        train_splits = (aachen_path / "train.uttlist").read_text().splitlines()
        validation_splits = (
            (aachen_path / "validation.uttlist").read_text().splitlines()
        )
        test_splits = (aachen_path / "test.uttlist").read_text().splitlines()

        data_train = ds.data[ds.data["img_id"].isin(train_splits)]
        data_eval = ds.data[ds.data["img_id"].isin(validation_splits)]
        data_test = ds.data[ds.data["img_id"].isin(test_splits)]

        ds_train = copy(ds)
        ds_train.data = data_train
        ds_train.set_transforms_for_split("train")

        ds_eval = copy(ds)
        ds_eval.data = data_eval
        ds_eval.set_transforms_for_split("eval")

        ds_test = copy(ds)
        ds_test.data = data_test
        ds_test.set_transforms_for_split("test")
    else:
        ds_train, ds_eval = torch.utils.data.random_split(
            ds, [math.ceil(0.8 * len(ds)), math.floor(0.2 * len(ds))]
        )
        ds_eval.dataset = copy(ds)
        ds_eval.dataset.set_transforms_for_split("eval")

    # Initialize dataloaders.
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_eval = DataLoader(
        ds_eval,
        batch_size=2 * args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.validate is not None:
        assert Path(
            args.validate
        ).is_file(), f"{args.validate} does not point to a file."
        # Load the model. Note that the vocab length and special tokens given below
        # are derived from the saved label encoder associated with the checkpoint.
        model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
            args.validate,
            label_encoder=ds.label_enc,
            vocab_len=len(ds.vocab),
            eos_tkn_idx=eos_tkn_idx,
            sos_tkn_idx=sos_tkn_idx,
            pad_tkn_idx=pad_tkn_idx,
        )
    else:
        model = LitFullPageHTREncoderDecoder(
            label_encoder=ds.label_enc,
            encoder_name=args.encoder,
            vocab_len=len(ds.vocab),
            d_model=args.d_model,
            max_seq_len=IAMDataset.MAX_SEQ_LENS[args.data_format],
            eos_tkn_idx=eos_tkn_idx,
            sos_tkn_idx=sos_tkn_idx,
            pad_tkn_idx=pad_tkn_idx,
            num_layers=args.num_layers,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            drop_enc=args.drop_enc,
            drop_dec=args.drop_dec,
            params_to_log={
                "batch_size": int(args.num_nodes * args.batch_size),
                "data_format": args.data_format,
                "seed": args.seed,
                "splits": ("Aachen" if args.use_aachen_splits else "random"),
                "max_epochs": args.max_epochs,
                "num_nodes": args.num_nodes,
                "precision": args.precision,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "early_stopping_patience": args.early_stopping_patience,
                "label_smoothing": args.label_smoothing,
            },
        )

    trainer = pl.Trainer(
        logger=tb_logger,
        strategy=(
            DDPPlugin(find_unused_parameters=False) if args.num_nodes != 1 else None
        ),  # ddp = Distributed Data Parallel
        precision=args.precision,  # default is 32 bit
        num_nodes=args.num_nodes,
        gpus=(0 if args.use_cpu else 1),
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        callbacks=[
            ModelCheckpoint(
                save_top_k=(-1 if args.save_all_checkpoints else 3),
                mode="min",
                monitor="char_error_rate",
                filename="{epoch}-{char_error_rate:.4f}-{word_error_rate:.4f}",
            ),
            LogModelPredictions(
                ds.label_enc,
                val_batch=next(
                    iter(
                        DataLoader(
                            Subset(
                                ds_eval,
                                random.sample(
                                    range(len(ds_eval)), LOGMODELPREDICTIONS_TO_SAMPLE
                                ),
                            ),
                            batch_size=LOGMODELPREDICTIONS_TO_SAMPLE,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=True,
                        )
                    )
                ),
                train_batch=next(
                    iter(
                        DataLoader(
                            Subset(
                                ds_train,
                                random.sample(
                                    range(len(ds_train)), LOGMODELPREDICTIONS_TO_SAMPLE
                                ),
                            ),
                            batch_size=LOGMODELPREDICTIONS_TO_SAMPLE,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=True,
                        )
                    )
                ),
                use_gpu=(False if args.use_cpu else True),
            ),
            EarlyStopping(
                monitor="char_error_rate",
                patience=args.early_stopping_patience,
                verbose=True,
                mode="min",
            ),
        ],
        # overfit_batches=1,
        # profiler="simple",  # set this to get a profiler report showing mean duration of function calls
    )

    if args.validate:  # validate a trained model
        trainer.validate(model, dl_eval)
    else:  # train a model
        trainer.fit(model, dl_train, dl_eval)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()

    # Model arguments.
    parser.add_argument("--encoder", type=str, choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=260)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--drop_enc", type=float, default=0.5, help="Encoder dropout.")
    parser.add_argument("--drop_dec", type=float, default=0.5, help="Decoder dropout.")

    # Trainer arguments.
    parser.add_argument("--max_epochs", type=int, default=999)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to train on.")
    parser.add_argument("--precision", type=int, default=16, help="How many bits of floating point precision to use.")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon (0.0 indicates no smoothing)")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--save_all_checkpoints", action="store_true", default=False)
    parser.add_argument("--use_cpu", action="store_true", default=False)

    # Program arguments.
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--data_format", type=str, choices=["form", "line", "word"], default="word")
    parser.add_argument("--use_aachen_splits", action="store_true", default=False)
    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validate", type=str, help="Validate a trained model, specified by its checkpoint path.")
    args = parser.parse_args()

    main(args)
    # fmt: on
