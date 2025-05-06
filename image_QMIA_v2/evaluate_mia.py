import os
import argparse
import shutil
import torch
import numpy as np
import random
from lightning_utils import LightningQMIA, CustomWriter
from data_utils import CustomDataModule
import pytorch_lightning as pl
import glob

def argparser():
    parser = argparse.ArgumentParser(description="QMIA evaluation")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
        help="image input size, set to -1 to use dataset's default value",
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

    print(f"\nRemoving {len(prediction_files)} original prediction files...")
    for pred_file in prediction_files:
        # Only remove files that match the pattern predictions_*.pt but not the aggregated files
        if os.path.basename(pred_file) not in ["predictions_test.pt", "predictions_val.pt"]:
            os.remove(pred_file)
    print("Original prediction files removed")

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
        "best_val_loss.ckpt"
    ))
    lightning_model.eval()

    datamodule = CustomDataModule(
        dataset_name=args.attack_dataset,
        stage=args.data_mode,
        num_workers=16,
        image_size=args.image_size,
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

    trainer.strategy.barrier()

    if trainer.strategy.is_global_zero:
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
        plot_roc_curve(test_preds_ood, val_preds_ood, test_label="private (ID)", val_label="public (ID)", title=f'(Kept Class/es, {kept_classes})', save_path="roc_curve_id")
    print("ROC curves plotted and saved.")

    print("Plotting scores per class...")
    plot_scores_per_class(test_preds, label="private", plot_type="cdf")
    plot_scores_per_class(val_preds, label="public", plot_type="cdf")
    plot_scores_per_class(test_preds, label="private", plot_type="violin")
    plot_scores_per_class(val_preds, label="public", plot_type="violin")
    print("Scores per class plotted and saved.")

    print("Analyzing statistical correlation between target scores and class labels...")
    stat_corr_scores_labels(test_preds, label="private")
    stat_corr_scores_labels(val_preds, label="public")

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
