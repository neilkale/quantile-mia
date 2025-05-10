import os
import argparse
import shutil
import sys
import torch
import numpy as np
import random

from tqdm import tqdm
from lightning_utils import LightningQMIA, CustomWriter
from data_utils import CustomDataModule
import pytorch_lightning as pl
import glob
import torch.distributed as dist

from matplotlib import pyplot as plt

def argparser():
    parser = argparse.ArgumentParser(description="QMIA evaluation")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=-1,
        help="image input size, set to -1 to use dataset's default value",
    )
    parser.add_argument(
        "--base_image_size",
        type=int,
        default=-1,
        help="image input size to base model, set to -1 to use dataset's default value",
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
        help="loss function (gaussian, pinball)",
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
        "--checkpoint",
        type=str,
        default="best_val_loss",
        help="checkpoint path (either best_val_loss or last)",
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
        "--results_root",
        type=str,
        default="./outputs/",
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        default="eval",
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
        "--rerun", action="store_true", help="whether to rerun the evaluation"
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

    args.attack_results_path = os.path.join(
        args.attack_checkpoint_path,
        "predictions",
    )

    args.attack_plots_path = os.path.join(
        args.attack_results_path,
        "plots",
    )

    if "cifar100" in args.base_model_dataset.lower():
        args.num_base_classes = 100
    elif "imagenet-1k" in args.base_model_dataset.lower():
        args.num_base_classes = 1000
    else:
        args.num_base_classes = 10

    return args

def aggregate_predictions(results_path):
    """
    Aggregate prediction files in results_path into two files:
    - predictions_test.pt: contains test set predictions
    - predictions_val.pt: contains validation set predictions
    
    Each file will contain a list of 5 elements:
    [pred_scores, target_scores, logits, targets, loss]
    where each element is a tensor containing all data for that quantity.
    """
    print(f"Aggregating predictions from {results_path}")
    
    # Get list of all prediction files
    prediction_files = sorted(glob.glob(os.path.join(results_path, "predictions_*.pt")))
    if not prediction_files:
        print(f"No prediction files found in {results_path}")
        return
    
    print(f"Found {len(prediction_files)} prediction files")
    
    # Initialize lists to collect results
    test_predictions = [[] for _ in range(5)]  # 5 elements from predict_step
    val_predictions = [[] for _ in range(5)]   # 5 elements from predict_step
    
    # Load and aggregate predictions
    for pred_file in prediction_files:
        print(f"Processing {os.path.basename(pred_file)}")
        batch_preds = torch.load(pred_file)
        
        # First element is test, second is val
        test_batch = batch_preds[0]
        val_batch = batch_preds[1]
        
        # For each batch in test_batch, extract the 5 elements
        for batch in test_batch:
            for i in range(5):
                test_predictions[i].append(batch[i])
        
        # For each batch in val_batch, extract the 5 elements
        for batch in val_batch:
            for i in range(5):
                val_predictions[i].append(batch[i])
    
    # Concatenate tensors for each element
    test_result = []
    val_result = []
    
    for i in range(5):
        if test_predictions[i]:  # Check if list is not empty
            # Concatenate along the first dimension (batch)
            test_result.append(torch.cat(test_predictions[i], dim=0))
        else:
            test_result.append(torch.tensor([]))
            
        if val_predictions[i]:  # Check if list is not empty
            # Concatenate along the first dimension (batch)
            val_result.append(torch.cat(val_predictions[i], dim=0))
        else:
            val_result.append(torch.tensor([]))
    
    # Save aggregated results
    torch.save(test_result, os.path.join(results_path, "predictions_test.pt"))
    torch.save(val_result, os.path.join(results_path, "predictions_val.pt"))
    
    print(f"Saved aggregated results to {results_path}/predictions_test.pt and {results_path}/predictions_val.pt")

def evaluate_mia(args, rerun=False):
    if os.path.exists(args.attack_results_path) and not rerun:
        print(f"Results already exist at {args.attack_results_path}.")
        return
    else:
        # Remove the existing results directory if it exists
        if os.path.exists(args.attack_results_path):
            print(f"Removing existing results directory at {args.attack_results_path}.")
            shutil.rmtree(args.attack_results_path)
        # Create a new results directory
        print(f"Creating results directory at {args.attack_results_path}.")
        os.makedirs(args.attack_results_path, exist_ok=True)
    
    # Create lightning model
    
    lightning_model = LightningQMIA.load_from_checkpoint(os.path.join(
        args.attack_checkpoint_path,
        f"{args.checkpoint}.ckpt"
    ))
    lightning_model.eval()

    datamodule = CustomDataModule(
        dataset_name=args.base_dataset,
        stage=args.data_mode,
        num_workers=16,
        image_size=args.image_size,
        base_image_size=args.base_image_size,
        batch_size=args.batch_size if not args.DEBUG else 2,
        data_root=args.data_root,
    )
    
    pred_writer = CustomWriter(
            output_dir=args.attack_results_path,
            write_interval="epoch",
        )
    trainer = pl.Trainer(
        accelerator="gpu" if not args.DEBUG else "cpu",
        devices=-1 if not args.DEBUG else 1,
        callbacks=[pred_writer],
        strategy="ddp",
        enable_progress_bar=True,
    )

    trainer.predict(
        model=lightning_model,
        datamodule=datamodule,
        return_predictions=False,
    )

    if dist.is_initialized():
        dist.barrier()
        if dist.get_rank() != 0:
            print(f"Process {dist.get_rank()} exiting early")
            sys.exit(0)  # Only non-zero rank processes exit here

    print("Aggregating predictions...")
    aggregate_predictions(args.attack_results_path)

def plot_scores_per_class(
    preds,
    label: str = "private",
    plot_type: str = "cdf",
    log_x: bool = True,
    dropped = []
):
    """
    Plot per‑class score distributions for a set of predictions.

    Parameters
    ----------
    preds : tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float)
        (pred_scores, target_scores, logits, targets, loss) as returned by the
        attack pipeline. `pred_scores` is expected to be shaped [N, 1].
    label : str, optional
        Tag used in figure titles / filenames.
    plot_type : {'cdf', 'hist', 'violin'}, optional
        Type of plot to draw.
    save_dir : str | None, optional
        Directory to place PNGs (defaults to ``args.attack_plots_path`` if unset).
    log_x : bool, optional
        Apply log‑scale to the x‑axis (ignored for violin).
    """
    pred_scores, tgt_scores, *_ , targets, _ = preds
    pred_scores = pred_scores[:, 0].cpu().numpy()
    tgt_scores = tgt_scores.cpu().numpy()
    targets = targets.cpu().numpy()
    classes = sorted(np.unique(targets))
    
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    for (i, cls) in enumerate(classes):
        cls_scores = pred_scores[targets == cls].ravel()
        axs[i].hist(cls_scores, bins=30, alpha=0.7)
        axs[i].set_title(f'Class: {i}')
        axs[i].set_xlabel('Predicted Mean')
        axs[i].set_ylabel('Frequency')
        
    plt.tight_layout()
    plt.savefig('plots_pt/threshold_prediction_histograms_{}.png'.format(''.join([str(x) for x in dropped])))

if __name__ == "__main__":
    args = argparser()

    print("Predicting on evaluation set...")
    evaluate_mia(
        args,
        rerun=args.rerun
    )

    test_preds = torch.load(os.path.join(args.attack_results_path, "predictions_test.pt"))
    val_preds = torch.load(os.path.join(args.attack_results_path, "predictions_val.pt"))

    if not os.path.exists(args.attack_plots_path):
        os.makedirs(args.attack_plots_path, exist_ok=True)

    plot_scores_per_class(test_preds, label="private", plot_type="cdf", dropped=args.cls_drop)
