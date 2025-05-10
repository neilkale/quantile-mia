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
    
    # Print some stats
    print("\nTest predictions:")
    for i, name in enumerate(["pred_scores", "target_scores", "logits", "targets", "loss"]):
        if len(test_result[i]) > 0:
            print(f"  {name}: shape {test_result[i].shape}")
    
    print("\nValidation predictions:")
    for i, name in enumerate(["pred_scores", "target_scores", "logits", "targets", "loss"]):
        if len(val_result[i]) > 0:
            print(f"  {name}: shape {val_result[i].shape}")

    # print(f"\nRemoving {len(prediction_files)} original prediction files...")
    # for pred_file in prediction_files:
    #     # Only remove files that match the pattern predictions_*.pt but not the aggregated files
    #     if os.path.basename(pred_file) not in ["predictions_test.pt", "predictions_val.pt"]:
    #         os.remove(pred_file)
    # print("Original prediction files removed")

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
        dataset_name=args.attack_dataset,
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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.stats as ss

def tpr_at_fpr(tprs, fprs, target_fpr):
    """
    Get the TPR at a specific FPR from the ROC curve data.
    """
    # Find the index of the closest FPR to the target FPR
    idx = np.argmin(np.abs(fprs - target_fpr))
    return tprs[idx]

def plot_roc_curve(test_preds, val_preds, test_label="private", val_label="public", title="", save_path="roc_curve"):
    test_pred_scores, test_target_scores = test_preds[0], test_preds[1]
    val_pred_scores, val_target_scores = val_preds[0], val_preds[1]

    # Compute z-scores
    test_mu, test_log_std = test_pred_scores[:, 0], test_pred_scores[:, 1]
    test_std = torch.exp(test_log_std)
    test_z_scores = (test_target_scores - test_mu) / test_std
    test_z_scores = test_z_scores.cpu().numpy()

    val_mu, val_log_std = val_pred_scores[:, 0], val_pred_scores[:, 1]
    val_std = torch.exp(val_log_std)
    val_z_scores = (val_target_scores - val_mu) / val_std
    val_z_scores = val_z_scores.cpu().numpy()

    # Compute ROC curve and ROC area for our method
    z_scores = np.concatenate((test_z_scores, val_z_scores))
    labels = np.concatenate((np.ones(len(test_z_scores)), np.zeros(len(val_z_scores))))
    fpr, tpr, _ = roc_curve(labels, z_scores)
    roc_auc = auc(fpr, tpr)

    # Compute ROC curve and ROC area for baseline
    scores = np.concatenate((test_target_scores.cpu().numpy(), val_target_scores.cpu().numpy()))
    baseline_fpr, baseline_tpr, _ = roc_curve(labels, scores)
    baseline_roc_auc = auc(baseline_fpr, baseline_tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='steelblue', label=f'Quantile MIA (AUC = {roc_auc:.2f})')
    plt.plot(baseline_fpr, baseline_tpr, color='indianred', label=f'Baseline (AUC = {baseline_roc_auc:.2f})')
    plt.plot([1e-4, 1], [1e-4, 1], color='gray', linestyle='--')
    plt.xlim([1e-4, 1.0])
    plt.ylim([1e-4, 1.0])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {title}')
    plt.legend(loc='lower right')

    plt.annotate(
        f'QMIA FPR at TPR=0.1%: {tpr_at_fpr(tpr, fpr, 1e-3)*100:.2f}%',
        xy=(0.01, 0.9), xycoords='axes fraction', color='steelblue',
    )
    plt.annotate(
        f'QMIA FPR at TPR=1%: {tpr_at_fpr(tpr, fpr, 1e-2)*100:.2f}%',
        xy=(0.01, 0.95), xycoords='axes fraction', color='steelblue',
    )

    plt.annotate(
        f'Baseline FPR at TPR=0.1%: {tpr_at_fpr(baseline_tpr, baseline_fpr, 1e-3)*100:.2f}%',
        xy=(0.01, 0.8), xycoords='axes fraction', color='indianred',
    )
    plt.annotate(
        f'Baseline FPR at TPR=1%: {tpr_at_fpr(baseline_tpr, baseline_fpr, 1e-2)*100:.2f}%',
        xy=(0.01, 0.85), xycoords='axes fraction', color='indianred',
    )

    plt.savefig(os.path.join(args.attack_plots_path, f"{save_path}.png"))
    plt.close()

def plot_roc_curves(test_preds, val_preds, labels=['ID', 'OOD']):
    plt.figure()

    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    for i, label in enumerate(labels):
        test_pred_scores, test_target_scores = test_preds[i][0], test_preds[i][1]
        val_pred_scores, val_target_scores = val_preds[i][0], val_preds[i][1]

        # Compute z-scores
        test_mu, test_log_std = test_pred_scores[:, 0], test_pred_scores[:, 1]
        test_std = torch.exp(test_log_std)
        test_z_scores = (test_target_scores - test_mu) / test_std
        test_z_scores = test_z_scores.cpu().numpy()

        val_mu, val_log_std = val_pred_scores[:, 0], val_pred_scores[:, 1]
        val_std = torch.exp(val_log_std)
        val_z_scores = (val_target_scores - val_mu) / val_std
        val_z_scores = val_z_scores.cpu().numpy()

        # Compute ROC curve and ROC area for our method
        z_scores = np.concatenate((test_z_scores, val_z_scores))
        labels = np.concatenate((np.ones(len(test_z_scores)), np.zeros(len(val_z_scores))))
        fpr, tpr, _ = roc_curve(labels, z_scores)
        roc_auc = auc(fpr, tpr)

        # Compute ROC curve and ROC area for baseline
        scores = np.concatenate((test_target_scores.cpu().numpy(), val_target_scores.cpu().numpy()))
        baseline_fpr, baseline_tpr, _ = roc_curve(labels, scores)
        baseline_roc_auc = auc(baseline_fpr, baseline_tpr)

        plt.plot(fpr, tpr, label=f'{label} Quantile MIA (AUC = {roc_auc:.2f})', color=colors[i])
        plt.annotate(
        f'QMIA FPR at TPR=0.1%: {tpr_at_fpr(tpr, fpr, 1e-3)*100:.2f}%',
        xy=(0.01, 0.9-0.1*i), xycoords='axes fraction', color=colors[i],
        )
        plt.annotate(
            f'QMIA FPR at TPR=1%: {tpr_at_fpr(tpr, fpr, 1e-2)*100:.2f}%',
            xy=(0.01, 0.95-0.1*i), xycoords='axes fraction', colors=colors[i],
        )
    
    plt.plot(baseline_fpr, baseline_tpr, color='darkgray', label=f'Baseline (AUC = {baseline_roc_auc:.2f})')
    plt.plot([1e-4, 1], [1e-4, 1], color='gray', linestyle='--')
    plt.xlim([1e-4, 1.0])
    plt.ylim([1e-4, 1.0])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    plt.annotate(
        f'Baseline FPR at TPR=0.1%: {tpr_at_fpr(baseline_tpr, baseline_fpr, 1e-3)*100:.2f}%',
        xy=(0.01, 0.8), xycoords='axes fraction', color='darkgray',
    )
    plt.annotate(
        f'Baseline FPR at TPR=1%: {tpr_at_fpr(baseline_tpr, baseline_fpr, 1e-2)*100:.2f}%',
        xy=(0.01, 0.85), xycoords='axes fraction', color='darkgray',
    )

    plt.savefig(os.path.join(args.attack_plots_path, "roc_curve.png"))
    plt.close()

def plot_scores_per_class(
    preds,
    label: str = "private",
    plot_type: str = "cdf",
    log_x: bool = True,
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
    classes = np.unique(targets)

    def _single_plot(scores, tag, colour):
        plt.figure(figsize=(10, 6))
        if plot_type == "violin":
            data = [scores[targets == cls].ravel() for cls in classes]
            parts = plt.violinplot(
                data,
                positions=np.arange(len(classes)),
                showmeans=True,
                showmedians=True,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(colour)
                pc.set_edgecolor("black")
                pc.set_alpha(0.7)
            parts["cmeans"].set_color("red")
            parts["cmedians"].set_color("black")
            plt.xticks(range(len(classes)), [f"Class {c}" for c in classes])
            plt.xlabel("Class")
            plt.ylabel("Score")
        else:
            for cls in classes:
                cls_scores = scores[targets == cls].ravel()
                if plot_type == "cdf":
                    x = np.sort(cls_scores)
                    y = np.arange(1, len(x) + 1) / len(x)
                    plt.plot(x, y, label=f"Class {cls}")
                elif plot_type == "hist":
                    plt.hist(
                        cls_scores,
                        bins=100,
                        alpha=0.5,
                        density=True,
                        label=f"Class {cls}",
                    )
            plt.xlabel("Score")
            plt.ylabel("Cumulative Prob." if plot_type == "cdf" else "Density")
        title = f"{plot_type.upper()} of {tag.capitalize()} Scores per Class – {label}"
        plt.title(title)
        if plot_type != "violin" and log_x:
            plt.xscale("log")
        plt.grid(True, linestyle="--", alpha=0.7, axis="y" if plot_type == "violin" else "both")
        if plot_type != "violin":
            plt.legend()
        plt.tight_layout()
        fname = f"{plot_type}_{tag}_scores_per_class_{label}.png"
        plt.savefig(os.path.join(args.attack_plots_path, fname))
        plt.close()

    _single_plot(pred_scores, "pred", "#3274A1")
    _single_plot(tgt_scores, "target", "#E1812C")

def class_effect_stats(score, y):
    groups = [score[y == c] for c in np.unique(y)]
    F, p_anova = ss.f_oneway(*groups)                 # parametric
    H, p_kw    = ss.kruskal(*groups)                  # non‑parametric
    
    # ETA‑squared (effect size)
    grand_mean = score.mean()
    sst = ((score - grand_mean)**2).sum()
    ssb = sum(len(g)*(g.mean() - grand_mean)**2 for g in groups)
    eta2 = ssb / sst
    
    return {'F':F, 'p_ANOVA':p_anova, 'H':H, 'p_KW':p_kw, 'eta2':eta2}

def stat_corr_scores_labels(preds, label: str = "private"):
    pred_scores, tgt_scores, *_, targets, _ = preds
    pred_scores = pred_scores[:, 0].cpu().numpy()      # predicted top‑two margin
    tgt_scores  = tgt_scores.cpu().numpy()             # top‑two margin
    targets     = targets.cpu().numpy()                # int labels 0..9

    stats_pred = class_effect_stats(pred_scores, targets)
    stats_tgt  = class_effect_stats(tgt_scores,  targets)

    print(f"\n=== {label.upper()}  SCORES vs. CLASS LABEL ===")
    print("Mean PREDICTED score (top two margin / 'confidence'):")
    print(f"  ANOVA:   F = {stats_pred['F']:.3f},  p = {stats_pred['p_ANOVA']:.3g}")
    print(f"  Kruskal: H = {stats_pred['H']:.3f},  p = {stats_pred['p_KW']:.3g}")
    print(f"  η² (effect size) = {stats_pred['eta2']:.3f}")

    print("\nTARGET score (top two margin / 'confidence'):")
    print(f"  ANOVA:   F = {stats_tgt['F']:.3f},  p = {stats_tgt['p_ANOVA']:.3g}")
    print(f"  Kruskal: H = {stats_tgt['H']:.3f},  p = {stats_tgt['p_KW']:.3g}")
    print(f"  η² (effect size) = {stats_tgt['eta2']:.3f}")

    return stats_pred, stats_tgt
    
from scipy.ndimage import sobel

def calculate_edge_density(image_tensor):
    image_np = image_tensor.cpu().numpy()
    
    # Calculate edge magnitude using Sobel filters
    if image_np.shape[0] == 3:  # RGB image
        # Process each channel
        edge_magnitude = 0
        for channel in range(3):
            dx = sobel(image_np[channel], axis=0)
            dy = sobel(image_np[channel], axis=1)
            edge_magnitude += np.sqrt(dx**2 + dy**2)
        edge_magnitude /= 3  # Average over channels
    else:  # Grayscale image
        dx = sobel(image_np[0], axis=0)
        dy = sobel(image_np[0], axis=1)
        edge_magnitude = np.sqrt(dx**2 + dy**2)
    
    # Calculate edge density as mean of edge magnitude
    edge_density = np.mean(edge_magnitude)
    
    return edge_density

def calculate_shannon_entropy(image_tensor, num_bins=256):
    # Flatten image tensor
    image_flat = image_tensor.view(-1).cpu().numpy()
    # Create histogram
    hist, _ = np.histogram(image_flat, bins=num_bins, range=(0, 1), density=True)
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def calculate_contrast_measure(np_image):
    """
    Calculate global contrast measure of an image.
    
    Args:
        np_image: Numpy array of image with shape (C, H, W)
        
    Returns:
        float: Contrast measure
    """
    # If RGB, convert to grayscale by averaging channels
    if np_image.shape[0] == 3:
        grayscale = np.mean(np_image, axis=0)
    else:
        grayscale = np_image[0]
    
    # Calculate global contrast (max - min) / (max + min)
    min_val = np.min(grayscale)
    max_val = np.max(grayscale)
    
    if (max_val + min_val) == 0:
        return 0
    
    global_contrast = (max_val - min_val) / (max_val + min_val)
    
    # Calculate local contrast (standard deviation of pixel neighborhoods)
    from scipy.ndimage import uniform_filter
    
    # Local mean
    local_mean = uniform_filter(grayscale, size=3)
    
    # Local variance
    local_var = uniform_filter(grayscale**2, size=3) - local_mean**2
    
    # Local standard deviation
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Average local contrast
    avg_local_contrast = np.mean(local_std)
    
    # Combine global and local contrast (weighted sum)
    combined_contrast = 0.5 * global_contrast + 0.5 * avg_local_contrast
    
    return combined_contrast

def calculate_gradient_magnitude(np_image):
    """
    Calculate statistics of gradient magnitude.
    
    Args:
        np_image: Numpy array of image with shape (C, H, W)
        
    Returns:
        float: Mean gradient magnitude
    """
    from scipy.ndimage import sobel
    
    # If RGB, convert to grayscale by averaging channels
    if np_image.shape[0] == 3:
        grayscale = np.mean(np_image, axis=0)
    else:
        grayscale = np_image[0]
    
    # Calculate gradients
    dx = sobel(grayscale, axis=0)
    dy = sobel(grayscale, axis=1)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    
    # Calculate statistics
    mean_magnitude = np.mean(gradient_magnitude)
    
    return mean_magnitude

def calculate_spatial_frequency(np_image):
    """
    Calculate spatial frequency metric.
    
    Args:
        np_image: Numpy array of image with shape (C, H, W)
        
    Returns:
        float: Spatial frequency metric
    """
    from scipy.fftpack import fft2, fftshift
    
    # If RGB, convert to grayscale by averaging channels
    if np_image.shape[0] == 3:
        grayscale = np.mean(np_image, axis=0)
    else:
        grayscale = np_image[0]
    
    # Calculate 2D FFT
    fft = fft2(grayscale)
    fft_shift = fftshift(fft)
    
    # Calculate magnitude spectrum
    magnitude = np.abs(fft_shift)
    
    # Create distance matrix from center
    rows, cols = grayscale.shape
    center_row, center_col = rows // 2, cols // 2
    row_grid, col_grid = np.mgrid[:rows, :cols]
    distance_from_center = np.sqrt((row_grid - center_row)**2 + (col_grid - center_col)**2)
    
    # Calculate weighted average of frequency components
    # Higher weights for high-frequency components (far from center)
    weighted_sum = np.sum(magnitude * distance_from_center)
    total_magnitude = np.sum(magnitude)
    
    if total_magnitude == 0:
        return 0
        
    spatial_frequency = weighted_sum / total_magnitude
    
    return spatial_frequency

def calculate_lbp_stats(np_image):
    """
    Calculate Local Binary Pattern statistics.
    
    Args:
        np_image: Numpy array of image with shape (C, H, W)
        
    Returns:
        float: LBP entropy (measure of texture complexity)
    """
    from skimage.feature import local_binary_pattern
    
    # If RGB, convert to grayscale by averaging channels
    if np_image.shape[0] == 3:
        grayscale = np.mean(np_image, axis=0)
    else:
        grayscale = np_image[0]
    
    # Calculate LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(grayscale, n_points, radius, method='uniform')
    
    # Calculate histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, density=True)
    
    # Calculate entropy of LBP histogram
    non_zero_hist = hist[hist > 0]
    entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))
    
    return entropy

def calculate_jpeg_compression_ratio(np_image):
    """
    Calculate JPEG compression ratio as a measure of image complexity.
    
    Args:
        np_image: Numpy array of image with shape (C, H, W)
        
    Returns:
        float: Compression ratio
    """
    import io
    from PIL import Image
    
    # Convert numpy array to PIL Image
    if np_image.shape[0] == 3:
        # Convert from (C, H, W) to (H, W, C) for PIL
        pil_image = Image.fromarray((np.transpose(np_image, (1, 2, 0)) * 255).astype(np.uint8))
    else:
        pil_image = Image.fromarray((np_image[0] * 255).astype(np.uint8))
    
    # Save as uncompressed format (PNG)
    uncompressed_buffer = io.BytesIO()
    pil_image.save(uncompressed_buffer, format='PNG')
    uncompressed_size = len(uncompressed_buffer.getvalue())
    
    # Save as compressed format (JPEG)
    compressed_buffer = io.BytesIO()
    pil_image.save(compressed_buffer, format='JPEG', quality=90)
    compressed_size = len(compressed_buffer.getvalue())
    
    # Calculate compression ratio
    if compressed_size == 0:
        return 0
        
    compression_ratio = uncompressed_size / compressed_size
    
    return compression_ratio

def calculate_fractal_dimension(np_image):
    """
    Calculate fractal dimension using box-counting method.
    
    Args:
        np_image: Numpy array of image with shape (C, H, W)
        
    Returns:
        float: Fractal dimension
    """
    from skimage.measure import find_contours
    
    # If RGB, convert to grayscale by averaging channels
    if np_image.shape[0] == 3:
        grayscale = np.mean(np_image, axis=0)
    else:
        grayscale = np_image[0]
    
    # Find contours
    contours = find_contours(grayscale, 0.5)
    
    if not contours:
        return 0
        
    # Use largest contour
    contour = max(contours, key=len)
    
    # Box counting method
    # Get box sizes
    p = min(grayscale.shape)
    box_sizes = np.logspace(0.01, np.log10(p/2), num=10)
    box_sizes = np.unique(np.int32(box_sizes))
    
    # Calculate count for each box size
    counts = []
    for box_size in box_sizes:
        if box_size == 0:
            continue
            
        # Create grid
        grid_shape = np.ceil(np.array(grayscale.shape) / box_size).astype(int)
        grid = np.zeros(grid_shape)
        
        # Mark boxes that contain contour points
        for point in contour:
            grid_x = int(point[0] / box_size)
            grid_y = int(point[1] / box_size)
            if grid_x < grid_shape[0] and grid_y < grid_shape[1]:
                grid[grid_x, grid_y] = 1
        
        counts.append(np.sum(grid))
    
    if len(counts) < 2:
        return 0
        
    # Fit line to log-log plot of box count vs size
    coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    
    # Fractal dimension is the negative slope
    fractal_dimension = -coeffs[0]
    
    return fractal_dimension

def calculate_perceptual_hash(np_image):
    """
    Calculate perceptual hash of an image.
    
    Args:
        np_image: Numpy array of image with shape (C, H, W)
        
    Returns:
        str: Perceptual hash string
    """
    from PIL import Image
    import imagehash
    
    # Convert numpy array to PIL Image
    if np_image.shape[0] == 3:
        # Convert from (C, H, W) to (H, W, C) for PIL
        pil_image = Image.fromarray((np.transpose(np_image, (1, 2, 0)) * 255).astype(np.uint8))
    else:
        pil_image = Image.fromarray((np_image[0] * 255).astype(np.uint8))
    
    # Calculate perceptual hash
    p_hash = imagehash.phash(pil_image)
    
    return str(p_hash)

def calculate_hash_distances(all_hashes, targets_dict):
    """
    Calculate perceptual hash distances between images of the same class.
    
    Args:
        all_hashes: Dictionary of {index: hash_string}
        targets_list: List of target labels
        
    Returns:
        dict: Dictionary of {index: average_distance_to_same_class}
    """
    import imagehash
    
    # Group hashes by class
    class_hashes = {}
    for idx, hash_str in all_hashes.items():
        if idx not in targets_dict:
            continue
            
        target = targets_dict[idx]
        if target not in class_hashes:
            class_hashes[target] = []
        class_hashes[target].append((idx, hash_str))

    # Group hashes by class
    distances = {}
    for target, hashes in class_hashes.items():
        for idx1, hash_str1 in hashes:
            # For phash, we need to convert the string back to a hash object
            try:
                hash1 = imagehash.hex_to_hash(hash_str1)
            except ValueError:
                # If conversion fails, create a new hash object
                hash1 = imagehash.ImageHash(np.array([int(bit) for bit in hash_str1]))
            
            # Calculate distances to other images in same class
            class_distances = []
            for idx2, hash_str2 in hashes:
                if idx1 != idx2:
                    try:
                        hash2 = imagehash.hex_to_hash(hash_str2)
                    except ValueError:
                        hash2 = imagehash.ImageHash(np.array([int(bit) for bit in hash_str2]))
                        
                    distance = hash1 - hash2
                    class_distances.append(distance)
            
            # Store average distance
            if class_distances:
                distances[idx1] = np.mean(class_distances)
            else:
                distances[idx1] = 0
    
    return distances
    
    # Calculate average distance to same class for each image
    distances = {}
    for target, hashes in class_hashes.items():
        for idx1, hash_str1 in hashes:
            hash1 = imagehash.hex_to_hash(hash_str1)
            
            # Calculate distances to other images in same class
            class_distances = []
            for idx2, hash_str2 in hashes:
                if idx1 != idx2:
                    hash2 = imagehash.hex_to_hash(hash_str2)
                    distance = hash1 - hash2
                    class_distances.append(distance)
            
            # Store average distance
            if class_distances:
                distances[idx1] = np.mean(class_distances)
            else:
                distances[idx1] = 0
    
    return distances

def calculate_feature_distances(all_features):
    """
    Calculate feature space clustering metrics.
    
    Args:
        all_features: Dictionary of {index: feature_vector}
        
    Returns:
        dict: Dictionary of {index: feature_distance_metric}
    """
    from sklearn.metrics.pairwise import cosine_distances
    
    # Convert to array for distance calculation
    indices = list(all_features.keys())
    features = np.array([all_features[idx] for idx in indices])
    
    # Calculate pairwise distances
    distances = cosine_distances(features)
    
    # Calculate average distance for each sample
    avg_distances = np.mean(distances, axis=1)
    
    # Create result dictionary
    result = {idx: dist for idx, dist in zip(indices, avg_distances)}
    
    return result

def calculate_difficulty_metrics(data_loader, output_path, label='unlabelled', batch_size=64, num_workers=16):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize lists for all metrics
    edge_densities = []
    entropies = []
    contrast_measures = []
    gradient_magnitudes = []
    spatial_frequency_metrics = []
    lbp_stats = []
    jpeg_compression_ratios = []
    fractal_dimensions = []
    sample_indices = []
    
    # For feature space clustering and perceptual hash
    all_features = {}
    all_hashes = {}
    targets_dict = {}  # Store targets for each sample

    import torchvision
    
    # For feature space clustering, we'll use a simple CNN feature extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = torchvision.models.resnet18(pretrained=True).to(device)
    feature_extractor.fc = torch.nn.Identity()  # Remove classifier layer
    feature_extractor.eval()
    
    print(f"Calculating difficulty metrics for {len(data_loader.dataset)} samples...")
    
    for batch_idx, (images, targets, base_images) in tqdm(enumerate(data_loader)):        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data_loader.dataset))
        batch_indices = list(range(start_idx, end_idx))
        
        # Initialize batch metrics
        batch_edge_densities = []
        batch_entropies = []
        batch_contrast_measures = []
        batch_gradient_magnitudes = []
        batch_spatial_frequency_metrics = []
        batch_lbp_stats = []
        batch_jpeg_compression_ratios = []
        batch_fractal_dimensions = []
        
        # Calculate feature embeddings for clustering metrics
        with torch.no_grad():
            images_device = images.to(device)
            batch_features = feature_extractor(images_device).cpu().numpy()
            
            # Store features for each sample
            for i, idx in enumerate(batch_indices):
                all_features[idx] = batch_features[i]
                targets_dict[idx] = targets[i].item()  # Store target for each sample
        
        # Process each image in the batch
        for i, image in enumerate(images):
            # Convert to numpy for processing
            np_image = image.cpu().numpy()
            
            # 1. Edge Density (existing)
            edge_density = calculate_edge_density(image)
            batch_edge_densities.append(edge_density)
            
            # 2. Shannon Entropy (existing)
            entropy = calculate_shannon_entropy(image)
            batch_entropies.append(entropy)
            
            # 3. Contrast Measure
            contrast = calculate_contrast_measure(np_image)
            batch_contrast_measures.append(contrast)
            
            # 4. Gradient Magnitude Statistics
            gradient_magnitude = calculate_gradient_magnitude(np_image)
            batch_gradient_magnitudes.append(gradient_magnitude)
            
            # 5. Spatial Frequency Metric
            spatial_frequency = calculate_spatial_frequency(np_image)
            batch_spatial_frequency_metrics.append(spatial_frequency)
            
            # 6. Local Binary Patterns
            lbp_stat = calculate_lbp_stats(np_image)
            batch_lbp_stats.append(lbp_stat)
            
            # 7. JPEG Compression Ratio
            jpeg_ratio = calculate_jpeg_compression_ratio(np_image)
            batch_jpeg_compression_ratios.append(jpeg_ratio)
            
            # 8. Fractal Dimension
            fractal_dim = calculate_fractal_dimension(np_image)
            batch_fractal_dimensions.append(fractal_dim)
            
            # 9. Perceptual Hash - Store hashes to calculate distances later
            img_hash = calculate_perceptual_hash(np_image)
            all_hashes[batch_indices[i]] = img_hash
        
        # Extend all metric lists
        edge_densities.extend(batch_edge_densities)
        entropies.extend(batch_entropies)
        contrast_measures.extend(batch_contrast_measures)
        gradient_magnitudes.extend(batch_gradient_magnitudes)
        spatial_frequency_metrics.extend(batch_spatial_frequency_metrics)
        lbp_stats.extend(batch_lbp_stats)
        jpeg_compression_ratios.extend(batch_jpeg_compression_ratios)
        fractal_dimensions.extend(batch_fractal_dimensions)
        sample_indices.extend(batch_indices)
    
    # 10. Feature-space clustering metrics
    # Calculate distances between samples in feature space
    print("Calculating feature distances...")
    feature_distances = calculate_feature_distances(all_features)
    
    # 11. Calculate perceptual hash distances
    print("Calculating perceptual hash distances...")
    phash_distances = calculate_hash_distances(all_hashes, targets_dict)
    
    # Save all results
    all_results = {
        'indices': sample_indices,
        'edge_densities': edge_densities,
        'entropies': entropies,
        'contrast_measures': contrast_measures,
        'gradient_magnitudes': gradient_magnitudes,
        'spatial_frequency_metrics': spatial_frequency_metrics,
        'lbp_stats': lbp_stats,
        'feature_distances': feature_distances,
        'phash_distances': phash_distances,
        'jpeg_compression_ratios': jpeg_compression_ratios,
        'fractal_dimensions': fractal_dimensions
    }
    
    print("Saving results...")
    torch.save(all_results, os.path.join(output_path, f"difficulty_metrics_{label}.pt"))

def evaluate_difficulty(args, rerun=False):
    save_dir = os.path.join(args.attack_results_path, "difficulty_metrics_test.pt")
    if os.path.exists(save_dir) and not rerun:
        print(f"Results already exist at {args.attack_results_path}.")
        return
    else:
        pass
    
    # Create datamodule 
    datamodule = CustomDataModule(
        dataset_name=args.attack_dataset,
        stage=args.data_mode,
        num_workers=16,
        image_size=args.image_size,
        batch_size=args.batch_size if not args.DEBUG else 2,
        data_root=args.data_root,
    )
    datamodule.setup()

    # Get dataset
    test_dataloader = datamodule.predict_dataloader()[0]
    val_dataloader = datamodule.predict_dataloader()[1]
    
    # Calculate difficulty metrics
    calculate_difficulty_metrics(
        data_loader=test_dataloader,
        label='test',
        output_path=args.attack_results_path,
        batch_size=args.batch_size if not args.DEBUG else 2,
        num_workers=16 if not args.DEBUG else 2
    )

    calculate_difficulty_metrics(
        data_loader=val_dataloader,
        label='val',
        output_path=args.attack_results_path,
        batch_size=args.batch_size if not args.DEBUG else 2,
        num_workers=16 if not args.DEBUG else 2
    )

def stat_corr_diff_scores_labels(diff_metrics, preds, label="private"):
    """
    Analyze the relationship between class labels and all difficulty metrics.
    
    Args:
        diff_metrics (dict): Dictionary containing difficulty metrics
        preds (list): Prediction results
        label (str): Dataset label (e.g., 'private', 'public')
    """
    from scipy import stats
    import numpy as np
    
    # Extract metrics
    edge_densities = np.array(diff_metrics['edge_densities'])
    entropies = np.array(diff_metrics['entropies'])
    contrast_measures = np.array(diff_metrics['contrast_measures'])
    gradient_magnitudes = np.array(diff_metrics['gradient_magnitudes'])
    spatial_frequency_metrics = np.array(diff_metrics['spatial_frequency_metrics'])
    lbp_stats = np.array(diff_metrics['lbp_stats'])
    jpeg_compression_ratios = np.array(diff_metrics['jpeg_compression_ratios'])
    fractal_dimensions = np.array(diff_metrics['fractal_dimensions'])
    
    # Extract feature distances (only use indices that match our dataset)
    feature_distances = np.array([diff_metrics['feature_distances'].get(idx, 0) for idx in diff_metrics['indices']])
    
    # Extract perceptual hash distances (only use indices that match our dataset)
    phash_distances = np.array([diff_metrics['phash_distances'].get(idx, 0) for idx in diff_metrics['indices']])
    
    # Extract class labels and target scores
    targets = preds[3].cpu().numpy()  # int labels
    tgt_scores = preds[1].cpu().numpy()  # target scores
    
    # Get unique classes
    unique_classes = np.unique(targets)
    
    print(f"\n=== {label.upper()} METRICS BY CLASS ANALYSIS ===")
    
    # List of metrics to analyze
    metrics = [
        ("TARGET SCORES", tgt_scores),
        ("EDGE DENSITY", edge_densities),
        ("SHANNON ENTROPY", entropies),
        ("CONTRAST MEASURE", contrast_measures),
        ("GRADIENT MAGNITUDE", gradient_magnitudes),
        ("SPATIAL FREQUENCY", spatial_frequency_metrics),
        ("LBP STATS", lbp_stats),
        ("JPEG COMPRESSION RATIO", jpeg_compression_ratios),
        ("FRACTAL DIMENSION", fractal_dimensions),
        ("FEATURE DISTANCE", feature_distances),
        ("PERCEPTUAL HASH DISTANCE", phash_distances)
    ]
    
    # Store effect sizes for comparison
    all_effect_sizes = {}
    
    # Analyze each metric
    for metric_name, metric_values in metrics:
        # ANOVA test
        f_stat, p_val = stats.f_oneway(*[metric_values[targets == cls] for cls in unique_classes])
        
        # Kruskal-Wallis test
        kw_stat, p_val_kw = stats.kruskal(*[metric_values[targets == cls] for cls in unique_classes])
        
        # Calculate effect size (eta-squared)
        ss_total = np.sum((metric_values - np.mean(metric_values))**2)
        ss_between = np.sum([len(metric_values[targets == cls]) * 
                           (np.mean(metric_values[targets == cls]) - np.mean(metric_values))**2 
                           for cls in unique_classes])
        
        # Avoid division by zero
        if ss_total == 0:
            eta_squared = 0
        else:
            eta_squared = ss_between / ss_total
        
        # Store effect size
        all_effect_sizes[metric_name] = eta_squared
        
        # Print results
        print(f"\n{metric_name} vs CLASS:")
        print(f"  ANOVA: F = {f_stat:.3f}, p = {p_val:.3g}")
        print(f"  Kruskal-Wallis: H = {kw_stat:.3f}, p = {p_val_kw:.3g}")
        print(f"  Effect Size (η²): {eta_squared:.3f}")
    
    # Print sorted effect sizes
    print("\nEFFECT SIZE COMPARISON (η²) - SORTED:")
    for metric_name, effect_size in sorted(all_effect_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {metric_name}: η² = {effect_size:.3f}")
    
    return all_effect_sizes

def compute_accuracies(test_preds, val_preds):
    """
    Compute accuracy of predictions.
    
    Args:
        preds (torch.Tensor): Predicted labels
        targets (torch.Tensor): True labels
        
    Returns:
        float: Accuracy
    """
    test_logits, test_targets = test_preds[2], test_preds[3]
    val_logits, val_targets = val_preds[2], val_preds[3]
    test_preds = torch.argmax(test_logits, dim=1)
    val_preds = torch.argmax(val_logits, dim=1)
    test_acc = (test_preds == test_targets).float().mean().item()
    val_acc = (val_preds == val_targets).float().mean().item()

    save_dir = os.path.join(args.attack_results_path, "base_accuracies.txt")
    with open(save_dir, "w") as f:
        f.write("Train (Private) Accuracy: {:.4f}\n".format(test_acc))
        f.write("Test (Public) Accuracy: {:.4f}\n".format(val_acc))
    

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

    print("Plotting ROC curves...")
    plot_roc_curve(test_preds, val_preds)
    if args.cls_drop:
        # OOD
        test_mask = torch.tensor([label.item() in args.cls_drop for label in test_preds[3]])
        test_preds_ood = [pred[test_mask] for pred in test_preds]
        val_mask = torch.tensor([label.item() in args.cls_drop for label in val_preds[3]])
        val_preds_ood = [pred[val_mask] for pred in val_preds]
        plot_roc_curve(test_preds_ood, val_preds_ood, test_label="private (OOD)", val_label="public (OOD)", title=f'(Dropped Class/es, {args.cls_drop})', save_path="roc_curve_ood")
        # ID
        test_mask = torch.tensor([label.item() not in args.cls_drop for label in test_preds[3]])
        test_preds_ood = [pred[test_mask] for pred in test_preds]
        val_mask = torch.tensor([label.item() not in args.cls_drop for label in val_preds[3]])
        val_preds_ood = [pred[val_mask] for pred in val_preds]
        all_classes = np.arange(args.num_base_classes)
        kept_classes = np.setdiff1d(all_classes, args.cls_drop)
        plot_roc_curve(test_preds_ood, val_preds_ood, test_label="private (ID)", val_label="public (ID)", title=f'(Kept Class/es, all but {args.cls_drop})', save_path="roc_curve_id")
    print("ROC curves plotted and saved.")

    # print("Plotting scores per class...")
    # plot_scores_per_class(test_preds, label="private", plot_type="cdf")
    # plot_scores_per_class(val_preds, label="public", plot_type="cdf")
    # plot_scores_per_class(test_preds, label="private", plot_type="violin")
    # plot_scores_per_class(val_preds, label="public", plot_type="violin")
    # print("Scores per class plotted and saved.")

    print("Computing accuracies...")
    compute_accuracies(test_preds, val_preds)
    print("Accuracies computed and saved.")



    # print("Analyzing statistical correlation between target scores and class labels...")
    # stat_corr_scores_labels(test_preds, label="private")
    # stat_corr_scores_labels(val_preds, label="public")

    # print("Analyzing statistical correlation between difficulty metrics and class labels...")
    # evaluate_difficulty(
    #     args, 
    #     rerun=True
    # )

    # test_diff_metrics = torch.load(os.path.join(args.attack_results_path, "difficulty_metrics_test.pt"), weights_only=False)
    # val_diff_metrics = torch.load(os.path.join(args.attack_results_path, "difficulty_metrics_val.pt"), weights_only=False)

    # stat_corr_diff_scores_labels(test_diff_metrics, test_preds, label="private")
    # stat_corr_diff_scores_labels(val_diff_metrics, val_preds, label="public")

# def plot_cdf_gap(test_preds, val_preds):
#     # Get target scores and class labels for test and validation sets
#     test_targets = test_preds[1].cpu().numpy().flatten()
#     test_labels = test_preds[3].cpu().numpy().flatten()
#     val_targets = val_preds[1].cpu().numpy().flatten()
#     val_labels = val_preds[3].cpu().numpy().flatten()

#     # Get union of classes from test and validation sets
#     unique_classes = np.union1d(np.unique(test_labels), np.unique(val_labels))

#     plt.figure(figsize=(10, 6))
#     for cls in unique_classes:
#         # Filter target scores for the current class
#         test_class_targets = test_targets[test_labels == cls]
#         val_class_targets = val_targets[val_labels == cls]

#         if len(test_class_targets) == 0 or len(val_class_targets) == 0:
#             continue

#         # Sort the scores
#         test_sorted = np.sort(test_class_targets)
#         val_sorted = np.sort(val_class_targets)

#         # Create a common grid for x-axis between the min and max of both arrays
#         xmin = min(test_sorted[0], val_sorted[0])
#         xmax = max(test_sorted[-1], val_sorted[-1])
#         x_grid = np.linspace(xmin, xmax, 1000)

#         # Compute empirical CDF for test and validation targets
#         test_cdf = np.searchsorted(test_sorted, x_grid, side='right') / len(test_sorted)
#         val_cdf = np.searchsorted(val_sorted, x_grid, side='right') / len(val_sorted)

#         # Compute the gap between the two CDFs
#         cdf_gap = test_cdf - val_cdf

#         # Plot the CDF gap for this class on the same plot
#         plt.plot(x_grid, cdf_gap, label=f'Class {int(cls)}')

#     plt.xlabel('Target Score')
#     plt.ylabel('CDF Gap (Private - Public)')
#     plt.title('Difference between CDF of Target Scores (Private vs. Public)')
#     plt.legend(title="Classes")
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.savefig(os.path.join(args.attack_plots_path, "cdf_gap_all.png"))
#     plt.close()
