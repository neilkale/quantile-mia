import argparse
import os
import shutil

import torch
import numpy as np
import random

from data_utils import CustomDataModule

import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning_utils import LightningBaseNet

def argparser():
    """
    Parse command line arguments for base model trainer.
    """
    parser = argparse.ArgumentParser(description="Base network trainer")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--min_factor",
        type=float,
        default=0.3,
        help="minimum learning rate factor for linear/cosine scheduler",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="l2 regularization"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--image_size",
        type=int,
        default=-1,
        help="image input size, set to -1 to use dataset's default value",
    )
    parser.add_argument(
        "--architecture", type=str, default="cifar-resnet-50", help="Model Type "
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer")
    parser.add_argument(
        "--scheduler", type=str, default="step", help="learning rate scheduler"
    )
    parser.add_argument(
        "--scheduler_step_gamma",
        type=float,
        default=0.2,
        help="scheduler reduction fraction for step scheduler",
    )
    parser.add_argument(
        "--scheduler_step_fraction",
        type=float,
        default=0.3,
        help="scheduler fraction of steps between decays",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=0.0, help="gradient clipping"
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="label_smoothing"
    )
    parser.add_argument("--dataset", type=str, default="cinic10/0_16", help="dataset")
    parser.add_argument(
        "--model_root",
        type=str,
        default="./models/",
        help="model directory",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/",
        help="dataset root directory",
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        default="base",
        help="data mode, either base, mia, or eval",
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

    args.base_checkpoint_path = os.path.join(
        args.model_root,
        "base",
        args.dataset,
        args.architecture
    )

    # Set number of base classes.
    if "cifar100" in args.dataset.lower():
        args.num_base_classes = 100
    elif "imagenet-1k" in args.dataset.lower():
        args.num_base_classes = 1000
    else:
        args.num_base_classes = 10

    # Set random seed.
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    return args

def train_model(config, args, callbacks=None, rerun=False):
    """
    Pretrain a classification model on a dataset to use as a model to run a QMIA attack on.
    """
    callbacks = callbacks or []
    save_handle = "model.pickle"
    checkpoint_path = os.path.join(args.base_checkpoint_path, save_handle)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if (
        os.path.exists(checkpoint_path)
        and not rerun
    ):
        print(f"Checkpoint already exists at {checkpoint_path}. Skipping base model training.")
        return
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    datamodule = CustomDataModule(
        dataset_name=args.dataset,
        num_workers=16,
        image_size=args.image_size,
        batch_size=args.batch_size,
        data_root=args.data_root,
        stage=args.data_mode,
    )
    
    lightning_model = LightningBaseNet(
        architecture=args.architecture,
        num_classes=args.num_base_classes,
        optimizer_params=config,
        label_smoothing=config["label_smoothing"],
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="ptl/val_acc1",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
        filename="best",
        enable_version_counter=False,
    )
    callbacks = callbacks + [checkpoint_callback] + [TQDMProgressBar()]

    trainer = pl.Trainer(
        max_epochs=config["epochs"] if not args.DEBUG else 1,
        accelerator="gpu" if not args.DEBUG else "cpu", 
        callbacks=callbacks,
        devices=-1 if not args.DEBUG else 1,
        default_root_dir=checkpoint_dir,
        strategy='fsdp' if not args.DEBUG else 'ddp',
        gradient_clip_val=config["gradient_clip_val"],
        log_every_n_steps=10,
    )

    torch.set_float32_matmul_precision('medium')
    trainer.logger.log_hyperparams(config)
    if trainer.global_rank == 0:
        print(args.dataset)
    trainer.fit(lightning_model, datamodule=datamodule)
    if trainer.global_rank == 0:
        # reload best network and save just the base model
        lightning_model = LightningBaseNet.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        torch.save(lightning_model.model.state_dict(), checkpoint_path)
        print(
            "saved model from {} to {} ".format(
                checkpoint_callback.best_model_path, checkpoint_path
            )
        )
    trainer.strategy.barrier()

if __name__ == "__main__":
    args = argparser()

    config = {
        "lr": args.lr,
        "scheduler": args.scheduler,
        "min_factor": args.min_factor,
        "epochs": args.epochs,
        "opt_type": args.optimizer,
        "weight_decay": args.weight_decay,
        "step_gamma": args.scheduler_step_gamma,
        "step_fraction": args.scheduler_step_fraction,
        "gradient_clip_val": args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
    }

    train_model(
        config,
        args,
        callbacks=None,
        rerun=args.rerun
    )
