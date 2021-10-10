import argparse
from functools import partial

from models import *
from data import IAMDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import GradientAccumulationScheduler


IAM_LOCATION = "/home/tobias/datasets/IAM"


def main(args):
    split = "train"
    parse_method = "form"

    seed_everything(args.seed)

    # TODO: use the Aachen splits
    ds = IAMDataset(args.data_dir, parse_method, split, use_cache=False)

    eos_tkn_idx, sos_tkn_idx, pad_tkn_idx = ds.label_enc.transform(
        [ds._eos_token, ds._sos_token, ds._pad_token]
    ).tolist()
    collate_fn = partial(
        IAMDataset.collate_fn, pad_val=pad_tkn_idx, eos_tkn_idx=eos_tkn_idx
    )

    ds_train, ds_eval = torch.utils.data.random_split(
        ds, [math.ceil(0.8 * len(ds)), math.floor(0.2 * len(ds))]
    )
    ds_eval.dataset.set_transforms_for_split("test")
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
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ds_train_debug = torch.utils.data.dataset.Subset(
        ds, [0, 1]
    )  # shortest target sequences
    dl_train_debug = DataLoader(
        ds_train_debug,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    encoder = FullPageHTREncoder(args.d_model, model_name="resnet18")
    decoder = FullPageHTRDecoder(
        vocab=ds.vocab,
        max_seq_len=args.max_seq_len,
        eos_tkn_idx=eos_tkn_idx,
        sos_tkn_idx=sos_tkn_idx,
        pad_tkn_idx=pad_tkn_idx,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
    )
    model = FullPageHTREncoderDecoder(encoder, decoder)

    # Use gradient accumulation, accumulating every 4 batches.
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # callbacks=[ShowPredictions(ds.label_enc)],
    )
    # trainer = pl.Trainer(gpus=1, precision=16, overfit_batches=1, max_epochs=2000)

    trainer.fit(model, dl_train, dl_eval)
    # trainer.fit(model, dl_train_debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=IAM_LOCATION)
    parser.add_argument("--max_epochs", type=int, default=999)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_seq_len", type=int, default=500
    )  # TODO: set this to a good value
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=260)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    main(args)
