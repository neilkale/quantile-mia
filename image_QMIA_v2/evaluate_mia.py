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
    parser = argparse.ArgumentParser(description="QMIA attack trainer")
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

def tpr_at_fpr(tprs, fprs, target_fpr):
    """
    Get the TPR at a specific FPR from the ROC curve data.
    """
    # Find the index of the closest FPR to the target FPR
    idx = np.argmin(np.abs(fprs - target_fpr))
    return tprs[idx]

def plot_roc_curve(test_preds, val_preds, test_label="private", val_label="public"):
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

    # Compute ROC curve and ROC area for test set
    z_scores = np.concatenate((test_z_scores, val_z_scores))
    labels = np.concatenate((np.ones(len(test_z_scores)), np.zeros(len(val_z_scores))))
    
    # Compute ROC curve and ROC area for validation set
    fpr, tpr, _ = roc_curve(labels, z_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='steelblue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([1e-4, 1], [1e-4, 1], color='gray', linestyle='--')
    plt.xlim([1e-4, 1.0])
    plt.ylim([1e-4, 1.0])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(args.attack_plots_path, "roc_curve.png"))
    plt.close()

def plot_scores_per_class(preds, label="private"):
    """
    Plot the scores per class for the given predictions.
    """
    pred_scores, target_scores, logits, targets, loss = preds
    pred_scores = pred_scores[:,0].cpu().numpy()
    target_scores = target_scores.cpu().numpy()
    targets = targets.cpu().numpy()

    # Get unique classes
    unique_classes = np.unique(targets)

    # Create a figure for the plot
    plt.figure(figsize=(10, 6))

    # Plot CDF for each class
    for cls in unique_classes:
        cls_mask = (targets == cls)
        # Get scores for this class
        scores = pred_scores[cls_mask]
        # Sort scores
        scores_sorted = np.sort(scores)
        # Create CDF (y-axis values from 0 to 1)
        cdf = np.arange(1, len(scores_sorted) + 1) / len(scores_sorted)
        # Plot CDF
        plt.plot(scores_sorted, cdf, label=f'Class {cls}')

    plt.xlabel('Score')
    plt.ylabel('Cumulative Probability')
    plt.title(f'CDF of Predicted Scores per Class - {label}')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.attack_plots_path, f"cdf_pred_scores_per_class_{label}.png"))
    plt.close()

    # Create a figure for the plot
    plt.figure(figsize=(10, 6))

    # Plot CDF for each class
    for cls in unique_classes:
        cls_mask = (targets == cls)
        # Get scores for this class
        scores = target_scores[cls_mask]
        # Sort scores
        scores_sorted = np.sort(scores)
        # Create CDF (y-axis values from 0 to 1)
        cdf = np.arange(1, len(scores_sorted) + 1) / len(scores_sorted)
        # Plot CDF
        plt.plot(scores_sorted, cdf, label=f'Class {cls}')

    plt.xlabel('Score')
    plt.ylabel('Cumulative Probability')
    plt.title(f'CDF of Target Scores per Class - {label}')
    plt.legend()
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.attack_plots_path, f"cdf_target_scores_per_class_{label}.png"))
    plt.close()

def plot_cdf_gap(test_preds, val_preds):
    # Get target scores and class labels for test and validation sets
    test_targets = test_preds[1].cpu().numpy().flatten()
    test_labels = test_preds[3].cpu().numpy().flatten()
    val_targets = val_preds[1].cpu().numpy().flatten()
    val_labels = val_preds[3].cpu().numpy().flatten()

    # Get union of classes from test and validation sets
    unique_classes = np.union1d(np.unique(test_labels), np.unique(val_labels))

    plt.figure(figsize=(10, 6))
    for cls in unique_classes:
        # Filter target scores for the current class
        test_class_targets = test_targets[test_labels == cls]
        val_class_targets = val_targets[val_labels == cls]

        if len(test_class_targets) == 0 or len(val_class_targets) == 0:
            continue

        # Sort the scores
        test_sorted = np.sort(test_class_targets)
        val_sorted = np.sort(val_class_targets)

        # Create a common grid for x-axis between the min and max of both arrays
        xmin = min(test_sorted[0], val_sorted[0])
        xmax = max(test_sorted[-1], val_sorted[-1])
        x_grid = np.linspace(xmin, xmax, 1000)

        # Compute empirical CDF for test and validation targets
        test_cdf = np.searchsorted(test_sorted, x_grid, side='right') / len(test_sorted)
        val_cdf = np.searchsorted(val_sorted, x_grid, side='right') / len(val_sorted)

        # Compute the gap between the two CDFs
        cdf_gap = test_cdf - val_cdf

        # Plot the CDF gap for this class on the same plot
        plt.plot(x_grid, cdf_gap, label=f'Class {int(cls)}')

    plt.xlabel('Target Score')
    plt.ylabel('CDF Gap (Private - Public)')
    plt.title('Difference between CDF of Target Scores (Private vs. Public)')
    plt.legend(title="Classes")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.attack_plots_path, "cdf_gap_all.png"))
    plt.close()

def plot_scores_violin_per_class(preds, label="private"):    
    pred_scores, target_scores, logits, targets, loss = preds
    pred_scores = pred_scores[:,0].cpu().numpy()
    target_scores = target_scores.cpu().numpy()
    targets = targets.cpu().numpy()

    # Get unique classes
    unique_classes = np.unique(targets)

    # Create a figure for the plot
    plt.figure(figsize=(10, 6))

    # Prepare data for violin plot - ensure 1D arrays
    data_to_plot = [np.ravel(pred_scores[targets == cls]) for cls in unique_classes]
    
    # Create violin plot
    violin_parts = plt.violinplot(data_to_plot, positions=range(len(unique_classes)), 
                                  showmeans=True, showmedians=True)
    
    # Customize violin plot appearance
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#3274A1')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customize mean and median lines
    violin_parts['cmeans'].set_color('red')
    violin_parts['cmedians'].set_color('black')
    
    # Set x-axis ticks and labels
    plt.xticks(range(len(unique_classes)), [f'Class {cls}' for cls in unique_classes])
    
    plt.xlabel('Class')
    plt.ylabel('Predicted Scores')
    plt.title(f'Violin Plot of Predicted Scores per Class - {label}')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.attack_plots_path, f"violin_pred_scores_per_class_{label}.png"))
    plt.close()

    # Create a figure for target scores
    plt.figure(figsize=(10, 6))

    # Prepare data for violin plot - ensure 1D arrays
    data_to_plot = [np.ravel(target_scores[targets == cls]) for cls in unique_classes]
    
    # Create violin plot
    violin_parts = plt.violinplot(data_to_plot, positions=range(len(unique_classes)), 
                                  showmeans=True, showmedians=True)
    
    # Customize violin plot appearance
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#E1812C')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Customize mean and median lines
    violin_parts['cmeans'].set_color('red')
    violin_parts['cmedians'].set_color('black')
    
    # Set x-axis ticks and labels
    plt.xticks(range(len(unique_classes)), [f'Class {cls}' for cls in unique_classes])
    
    plt.xlabel('Class')
    plt.ylabel('Target Scores')
    plt.title(f'Violin Plot of Target Scores per Class - {label}')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.attack_plots_path, f"violin_target_scores_per_class_{label}.png"))
    plt.close()

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

    print("Plotting ROC curve...")
    plot_roc_curve(test_preds, val_preds)
    print("ROC curve plotted and saved.")

    print("Plotting scores per class...")
    plot_scores_per_class(test_preds, label="private")
    plot_scores_per_class(val_preds, label="public")
    plot_scores_violin_per_class(test_preds, label="private")
    plot_scores_violin_per_class(val_preds, label="public")
    plot_cdf_gap(test_preds, val_preds)
    print("Scores per class plotted and saved.")

    
