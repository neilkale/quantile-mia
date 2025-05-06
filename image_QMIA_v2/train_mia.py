import argparse
import os
import time

import torch
import numpy as np
import random

from lightning_utils import LightningQMIA
from data_utils import CustomDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import pytorch_lightning as pl

def argparser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    
    parser = argparse.ArgumentParser(description="QMIA attack trainer")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument(
        "--epochs", type=int, default=30, help="epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="l2 regularization"
    )
    parser.add_argument(
        "--opt", type=str, default="adamw", help="optimizer (sgd, adam, adamw)"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="gradient clipping"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="image input size, set to -1 to use dataset's default value",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",   
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="",
        help="learning rate scheduler (step, cosine)",
    )
    
    parser.add_argument(
        "--architecture",
        type=str,
        default="facebook/convnext-tiny-224",
        help="Attack Model Type",
    )

    parser.add_argument(
        "--base_architecture",
        type=str,
        default="resnet-18",
        help="Base Model Type",
    )
    
    parser.add_argument(
        "--score_fn",
        type=str,
        default="top_two_margin",
        help="score function (true_logit_margin, top_two_margin)",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="gaussian",
        help="loss function (gaussian, quantile)",
    )

    parser.add_argument(
        "--base_model_dataset",
        type=str,
        default="cinic10/0_16",
        help="dataset (i.e. cinic10/0_16, imagenet/0_16, cifar100/0_16)",
    )
    parser.add_argument(
        "--attack_dataset",
        type=str,
        default=None,
        help="dataset (i.e. cinic10/0_16, imagenet/0_16, cifar100/0_16), if None, use the same as base_model_dataset",
    )


    parser.add_argument(
        "--model_root",
        type=str,
        default="./models/",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/",
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        default="mia",
        help="data mode (either base, mia, or eval)",
    )

    parser.add_argument(
        "--cls_drop",
        type=int,
        nargs="*",
        default=[],
        help="drop classes from the dataset, e.g. --cls_drop 1 3 7",
    )

    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="debug mode, set to True to run on CPU and with fewer epochs",
    )

    parser.add_argument(
        "--rerun",
        action="store_true",
        help="rerun training even if checkpoint exists",
    )

    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.attack_dataset is None:
        args.attack_dataset = args.base_model_dataset

    cls_drop_str = (
        "".join(str(c) for c in args.cls_drop)
        if args.cls_drop
        else "none"
    )
    
    args.attack_checkpoint_path = os.path.join(
        args.model_root,
        "mia",
        "base_" + args.base_model_dataset,
        args.base_architecture,
        "attack_" + args.attack_dataset,
        args.architecture,
        "score_fn_" + args.score_fn,
        "loss_fn_" + args.loss_fn,
        "cls_drop_" + cls_drop_str,
    )

    args.base_checkpoint_path = os.path.join(
        args.model_root,
        "base",
        args.base_model_dataset,
        args.base_architecture
    )

    if "cifar100" in args.base_model_dataset.lower():
        args.num_base_classes = 100
    elif "imagenet-1k" in args.base_model_dataset.lower():
        args.num_base_classes = 1000
    else:
        args.num_base_classes = 10

    return args

def train_model(args):
    start = time.time()

    metric = "ptl/val_loss"
    mode = "min"

    # Create lightning model
    lightning_model = LightningQMIA(
        architecture=args.architecture,
        base_architecture=args.base_architecture,
        image_size=args.image_size,
        hidden_dims=[512,512],
        num_classes=args.num_base_classes,
        optimizer_params={
            "opt_type": args.opt,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "epochs": args.epochs,
        },
        loss_fn=args.loss_fn,
        score_fn=args.score_fn,
        base_model_dir=args.base_checkpoint_path,
    )
    datamodule = CustomDataModule(
        dataset_name=args.attack_dataset,
        stage=args.data_mode,
        num_workers=16,
        image_size=args.image_size,
        batch_size=args.batch_size if not args.DEBUG else 2,
        data_root=args.data_root,
        cls_drop=args.cls_drop,
    )
    metric = "ptl/val_loss"
    mode = "min"
    checkpoint_dir = args.attack_checkpoint_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_val_loss",
        monitor=metric,
        mode=mode,
        save_top_k=1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    callbacks = [checkpoint_callback] + [TQDMProgressBar(10)]
    trainer = pl.Trainer(
        max_epochs=args.epochs if not args.DEBUG else 1,
        accelerator="gpu" if not args.DEBUG else "cpu", 
        callbacks=callbacks,
        devices=-1 if not args.DEBUG else 1,
        default_root_dir=checkpoint_dir,
        strategy='ddp' if not args.DEBUG else 'ddp',
        gradient_clip_val=args.grad_clip,
    )

    torch.set_float32_matmul_precision('medium')
    trainer.fit(lightning_model, datamodule=datamodule)

    print("Training finished in {:.2f} seconds".format(time.time() - start))
    print("Best checkpoint saved at {}".format(checkpoint_callback.best_model_path))

if __name__ == "__main__":
    args = argparser()

    if (
        os.path.exists(os.path.join(args.attack_checkpoint_path, "best_val_loss.ckpt"))
        and not args.rerun
    ):
        print(f"Checkpoint already exists at {args.attack_checkpoint_path}. Skipping attack model training.")
    else:
        os.makedirs(args.attack_checkpoint_path, exist_ok=True)
        train_model(args)

        

