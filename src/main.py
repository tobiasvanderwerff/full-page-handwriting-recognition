#!/usr/bin/env python3

import argparse
import random
import math
from copy import copy
from pathlib import Path
from functools import partial

from lit_models import LitFullPageHTREncoderDecoder
from lit_callbacks import LogModelPredictions, LogWorstPredictions, PREDICTIONS_TO_LOG
from data import IAMDataset, IAMDatasetSynthetic, IAMSyntheticDataGenerator
from util import LitProgressBar, LabelEncoder

import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.plugins import DDPPlugin

LOGGING_DIR = "lightning_logs/"


def main(args):

    seed_everything(args.seed)

    log_dir_root = Path(__file__).parent.parent.resolve()
    tb_logger = pl_loggers.TensorBoardLogger(
        str(log_dir_root / LOGGING_DIR), name="", version=args.experiment_name
    )

    label_enc = None
    n_classes_saved = None
    if args.validate or args.test or args.load_model:
        # Load the label encoder for the trained model.
        if args.validate:
            model_path = Path(args.validate)
        elif args.test:
            model_path = Path(args.test)
        else:
            model_path = Path(args.load_model)
        le_path_1 = model_path.parent.parent / "label_encoding.txt"
        le_path_2 = model_path.parent.parent / "label_encoder.pkl"
        assert le_path_1.is_file() or le_path_2.is_file(), (
            f"Label encoder file not found at {le_path_1} or {le_path_2}. "
            f"Make sure 'label_encoding.txt' exists in the lightning_logs directory."
        )
        le_path = le_path_2 if le_path_2.is_file() else le_path_1
        label_enc = LabelEncoder().read_encoding(le_path)
        n_classes_saved = label_enc.n_classes  # num. output classes for the saved model
        if args.data_format == "form":
            # Add the `\n` token to the label encoder (since forms can contain newlines)
            label_enc.add_classes(["\n"])

    ds = IAMDataset(
        args.data_dir,
        args.data_format,
        "train",
        label_enc=label_enc,
        only_lowercase=args.use_lowercase,
    )

    if n_classes_saved is None:
        n_classes_saved = ds.label_enc.n_classes

    if not args.validate:
        # Save the label encoder.
        save_dir = Path(tb_logger.log_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        ds.label_enc.dump(save_dir)

    eos_tkn_idx, sos_tkn_idx, pad_tkn_idx = ds.label_enc.transform(
        [ds._eos_token, ds._sos_token, ds._pad_token]
    )
    collate_fn = partial(
        IAMDataset.collate_fn, pad_val=pad_tkn_idx, eos_tkn_idx=eos_tkn_idx
    )

    # Split the dataset into train/val/(test).
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
        data_val = ds.data[ds.data["img_id"].isin(validation_splits)]
        data_test = ds.data[ds.data["img_id"].isin(test_splits)]

        ds_train = copy(ds)
        ds_train.data = data_train
        ds_train.set_transforms_for_split("train")

        ds_val = copy(ds)
        ds_val.data = data_val
        ds_val.set_transforms_for_split("val")

        ds_test = copy(ds)
        ds_test.data = data_test
        ds_test.set_transforms_for_split("test")
    else:
        ds_train, ds_val = torch.utils.data.random_split(
            ds, [math.ceil(0.8 * len(ds)), math.floor(0.2 * len(ds))]
        )
        ds_val.dataset = copy(ds)
        ds_val.dataset.set_transforms_for_split("val")

    worker_init_fn = None
    if args.synthetic_augmentation_proba > 0.0:
        words_per_sequence = (7, 13)
        if args.data_format == "line":
            # Change the length of the sampled IAM word sequences for more diversity
            # in writing style on a single line.
            words_per_sequence = (2, 5)
        ds_train = IAMDatasetSynthetic(
            ds_train,
            synth_prob=args.synthetic_augmentation_proba,
            words_per_sequence=words_per_sequence,
        )
        worker_init_fn = IAMSyntheticDataGenerator.get_worker_init_fn()

    # Initialize dataloaders.
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=2 * args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=2 * args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.validate or args.test or args.load_model:
        assert Path(model_path).is_file(), f"{model_path} does not point to a file."
        # Load the model. Note that the vocab length and special tokens given below
        # are derived from the saved label encoder associated with the checkpoint.
        model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
            str(model_path),
            label_encoder=ds.label_enc,
            vocab_len=n_classes_saved,
        )
        if args.load_model:
            model.model.set_num_output_classes(ds.label_enc.n_classes)
    else:
        model = LitFullPageHTREncoderDecoder(
            label_encoder=ds.label_enc,
            learning_rate=args.learning_rate,
            max_seq_len=IAMDataset.MAX_SEQ_LENS[args.data_format],
            d_model=args.d_model,
            num_layers=args.num_layers,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            encoder_name=args.encoder,
            drop_enc=args.drop_enc,
            drop_dec=args.drop_dec,
            label_smoothing=args.label_smoothing,
            params_to_log={
                "batch_size": int(args.num_nodes * args.batch_size)
                * (args.accumulate_grad_batches or 1),
                "data_format": args.data_format,
                "seed": args.seed,
                "splits": ("Aachen" if args.use_aachen_splits else "random"),
                "max_epochs": args.max_epochs,
                "num_nodes": args.num_nodes,
                "precision": args.precision,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "early_stopping_patience": args.early_stopping_patience,
                "label_smoothing": args.label_smoothing,
                "synthetic_augmentation_proba": args.synthetic_augmentation_proba,
                "gradient_clip_val": args.gradient_clip_val,
                "only_lowercase": args.use_lowercase,
                "loaded_model": args.load_model,
            },
        )

    callbacks = [
        ModelCheckpoint(
            save_top_k=(-1 if args.save_all_checkpoints else 3),
            mode="min",
            monitor="word_error_rate",
            filename="{epoch}-{char_error_rate:.4f}-{word_error_rate:.4f}",
        ),
        ModelSummary(max_depth=2),
        LitProgressBar(),
        LogWorstPredictions(
            dl_train,
            dl_val,
            dl_test,
            training_skipped=(args.validate is not None or args.test is not None),
            data_format=args.data_format,
        ),
        LogModelPredictions(
            ds.label_enc,
            val_batch=next(
                iter(
                    DataLoader(
                        Subset(
                            ds_val,
                            random.sample(
                                range(len(ds_val)), PREDICTIONS_TO_LOG[args.data_format]
                            ),
                        ),
                        batch_size=PREDICTIONS_TO_LOG[args.data_format],
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
                                range(len(ds_train)),
                                PREDICTIONS_TO_LOG[args.data_format],
                            ),
                        ),
                        batch_size=PREDICTIONS_TO_LOG[args.data_format],
                        shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                )
            ),
            data_format=args.data_format,
            use_gpu=(False if args.use_cpu else True),
        ),
    ]
    if args.early_stopping_patience != -1:
        callbacks.append(
            EarlyStopping(
                monitor="word_error_rate",
                patience=args.early_stopping_patience,
                verbose=True,
                mode="min",
                check_on_train_epoch_end=False,
            )
        )

    trainer = Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        strategy=(
            DDPPlugin(find_unused_parameters=False) if args.num_nodes != 1 else None
        ),  # ddp = Distributed Data Parallel
        gpus=(0 if args.use_cpu else 1),
        callbacks=callbacks,
    )

    if args.validate:  # validate a trained model
        trainer.validate(model, dl_val)
    elif args.test:  # test a trained model
        trainer.test(model, dl_test)
    else:  # train a model
        trainer.fit(model, dl_train, dl_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--data_format", type=str, choices=["form", "line", "word"],
                        default="word")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--synthetic_augmentation_proba", type=float, default=0.0,
                        help=("Probability of sampling synthetic IAM line/form images "
                              "during training."))
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon (0.0 indicates no smoothing)")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Start training from a saved model, specified by its "
                             "checkpoint path.")
    parser.add_argument("--validate", type=str, default=None,
                        help="Validate a trained model, specified by its checkpoint "
                             "path.")
    parser.add_argument("--test", type=str, default=None,
                        help="Test a trained model, specified by its checkpoint path.")
    parser.add_argument("--use_aachen_splits", action="store_true", default=False)
    parser.add_argument("--use_lowercase", action="store_true", default=False,
                        help="Convert all target label sequences to lowercase.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=-1,
                        help="Number of checks with no improvement after which "
                             "training will be stopped. Setting this to -1 will disable "
                             "early stopping.")
    parser.add_argument("--save_all_checkpoints", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name, used as the name of the folder in "
                             "which logs are stored.")
    # fmt: on

    parser = LitFullPageHTREncoderDecoder.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)  # adds Pytorch Lightning arguments

    args = parser.parse_args()

    main(args)
