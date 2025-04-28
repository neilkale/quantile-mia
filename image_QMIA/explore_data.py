from glob import glob
import os

import torch
from data_utils import CustomDataModule
from train_mia_ray import argparser

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import torch

# Function to plot the distribution of quantile thresholds by class
def plot_quantile_distributions(predicted_quantiles, targets, title="Quantile Threshold Distributions by Class", clip_range=None):
    """
    Plot distributions of quantile thresholds for each class
    
    Args:
        predicted_quantiles: Tensor of shape [n_samples, n_quantiles]
        targets: Tensor of shape [n_samples] with values 0-9
        title: Plot title
    """
    # Convert to numpy for easier handling
    if isinstance(predicted_quantiles, torch.Tensor):
        predicted_quantiles = predicted_quantiles.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
        
    # Handle -inf values (replace with minimum finite value)
    mask = np.isinf(predicted_quantiles)
    if np.any(mask):
        # Get the minimum non-inf value
        min_finite = np.min(predicted_quantiles[~mask]) if np.any(~mask) else -100
        # Replace -inf with this value
        predicted_quantiles[mask & (predicted_quantiles < 0)] = min_finite
    
    n_classes = 10  # For CINIC-10 dataset
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    
    # Create a colormap
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Plot histograms for each class
    for class_idx in range(n_classes):
        # Get quantiles for samples in this class
        class_mask = targets == class_idx
        if not np.any(class_mask):
            continue  # Skip if no samples for this class
            
        class_quantiles = predicted_quantiles[class_mask].reshape(-1)
        
        # Additional clipping if range is provided
        if clip_range is not None:
            min_val, max_val = clip_range
            class_quantiles = np.clip(class_quantiles, min_val, max_val)
        
        # Plot histogram for this class
        plt.hist(class_quantiles, bins=50, alpha=0.7, 
                 color=colors[class_idx], 
                 label=f'Class {class_idx}',
                 density=True)
    
    plt.xlabel('Quantile Threshold Values')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    return plt

# Function to create heatmap of average quantile values by class
def plot_quantile_heatmap(predicted_quantiles, targets, title="Average Quantile Values by Class"):
    """
    Create a heatmap showing average quantile values for each class
    
    Args:
        predicted_quantiles: Tensor of shape [n_samples, n_quantiles]
        targets: Tensor of shape [n_samples] with values 0-9
        title: Plot title
    """
    # Convert to numpy for easier handling
    if isinstance(predicted_quantiles, torch.Tensor):
        predicted_quantiles = predicted_quantiles.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    
    n_classes = 10  # For CINIC-10 dataset
    n_quantiles = predicted_quantiles.shape[1]
    
    # Initialize array to hold average quantile values for each class
    avg_quantiles = np.zeros((n_classes, n_quantiles))
    
    # Calculate average quantile values for each class
    for class_idx in range(n_classes):
        class_mask = targets == class_idx
        if np.any(class_mask):
            avg_quantiles[class_idx] = np.mean(predicted_quantiles[class_mask], axis=0)
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(avg_quantiles, cmap="viridis", 
                xticklabels=5, yticklabels=list(range(n_classes)),
                cbar_kws={'label': 'Average Quantile Value'})
    
    plt.xlabel('Quantile Index (0-40)')
    plt.ylabel('Class')
    plt.title(title)
    
    return plt

# Function to plot the quantile curves for each class
def plot_quantile_curves(predicted_quantiles, targets, title="Quantile Curves by Class"):
    """
    Plot median and quantile curves for each class
    
    Args:
        predicted_quantiles: Tensor of shape [n_samples, n_quantiles]
        targets: Tensor of shape [n_samples] with values 0-9
        title: Plot title
    """
    # Convert to numpy for easier handling
    if isinstance(predicted_quantiles, torch.Tensor):
        predicted_quantiles = predicted_quantiles.numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    
    n_classes = 10  # For CINIC-10 dataset
    n_quantiles = predicted_quantiles.shape[1]
    quantile_indices = np.arange(n_quantiles)
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Create a colormap
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Plot quantile curves for each class
    for class_idx in range(n_classes):
        class_mask = targets == class_idx
        if not np.any(class_mask):
            continue  # Skip if no samples for this class
        
        # Get quantile data for this class
        class_quantiles = predicted_quantiles[class_mask]
        
        # Calculate median and quantiles
        median = np.median(class_quantiles, axis=0)
        q25 = np.percentile(class_quantiles, 25, axis=0)
        q75 = np.percentile(class_quantiles, 75, axis=0)
        
        # Plot median line
        plt.plot(quantile_indices, median, color=colors[class_idx], 
                 label=f'Class {class_idx}', linewidth=2)
        
        # Plot quantile range as shaded area
        plt.fill_between(quantile_indices, q25, q75, color=colors[class_idx], alpha=0.2)
    
    plt.xlabel('Quantile Index (0-40)')
    plt.ylabel('Quantile Threshold Values')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    return plt

# Main function to generate all visualizations
def generate_qmia_visualizations(predicted_quantiles, targets, clip_range=None):
    """
    Generate multiple visualizations for QMIA analysis
    
    Args:
        predicted_quantiles: Tensor of shape [n_samples, n_quantiles]
        targets: Tensor of shape [n_samples] with values 0-9
    """
    # Generate all three plots
    hist_plot = plot_quantile_distributions(
        predicted_quantiles, targets, 
        title="Distribution of Quantile Threshold Values by Class",
        clip_range=clip_range
    )
    hist_plot.savefig('qmia_distribution_histogram.png')
    hist_plot.close()
    
    heatmap_plot = plot_quantile_heatmap(
        predicted_quantiles, targets,
        title="Average Quantile Values by Class"
    )
    heatmap_plot.savefig('qmia_quantile_heatmap.png')
    heatmap_plot.close()
    
    curves_plot = plot_quantile_curves(
        predicted_quantiles, targets,
        title="Quantile Curves by Class"
    )
    curves_plot.savefig('qmia_quantile_curves.png')
    curves_plot.close()
    
    print("All visualizations have been created!")

def compare_dropped_class(args):
    # datamodule = CustomDataModule(
    #     dataset_name=args.dataset,
    #     mode="mia",
    #     num_workers=6,
    #     image_size=args.image_size,
    #     batch_size=args.batch_size,
    #     data_root=args.data_root,
    #     cls_drop=args.cls_drop,
    # )
    # datamodule.setup()
    # train_loader = datamodule.train_dataloader()

    fig_name="best"

    prediction_output_dir = 'models/cinic10/0_16/mia/gaussian_qmia/facebook/convnext-large-224-22k-1k/use_hinge_True/use_target_label_False/use_target_inputs_False/cls_drop_/predictions/best'

    predict_results = None
    for file in glob(os.path.join(prediction_output_dir, "*.pt")):
        print(file)
        rank_predict_results = torch.load(file)
        if predict_results is None:
            predict_results = rank_predict_results
        else:
            for r, p in zip(rank_predict_results, predict_results):
                p.extend(r)

    def join_list_of_tuples(list_of_tuples):
        n_tuples = len(list_of_tuples[0])
        result = []
        for _ in range(n_tuples):
            try:
                result.append(torch.concat([t[_] for t in list_of_tuples]))
            except:
                result.append(torch.Tensor([t[_] for t in list_of_tuples]))
        return result

    (
        private_predicted_quantile_threshold,
        private_target_score,
        private_loss,
        private_base_acc1,
        private_base_acc5,
        private_targets
    ) = join_list_of_tuples(predict_results[-1])
    (
        test_predicted_quantile_threshold,
        test_target_score,
        test_loss,
        test_base_acc1,
        test_base_acc5,
        test_targets
    ) = join_list_of_tuples(predict_results[1])

    prediction_output_dir_drop0 = 'models/cinic10/0_16/mia/gaussian_qmia/facebook/convnext-large-224-22k-1k/use_hinge_True/use_target_label_False/use_target_inputs_False/cls_drop_0/predictions/best'

    predict_results = None
    for file in glob(os.path.join(prediction_output_dir_drop0, "*.pt")):
        rank_predict_results = torch.load(file)
        if predict_results is None:
            predict_results = rank_predict_results
        else:
            for r, p in zip(rank_predict_results, predict_results):
                p.extend(r)

    def join_list_of_tuples(list_of_tuples):
        n_tuples = len(list_of_tuples[0])
        result = []
        for _ in range(n_tuples):
            try:
                result.append(torch.concat([t[_] for t in list_of_tuples]))
            except:
                result.append(torch.Tensor([t[_] for t in list_of_tuples]))
        return result

    (
        private_predicted_quantile_threshold_drop0,
        private_target_score_drop0,
        private_loss_drop0,
        private_base_acc1_drop0,
        private_base_acc5_drop0,
        private_targets_drop0
    ) = join_list_of_tuples(predict_results[-1])
    (
        test_predicted_quantile_threshold_drop0,
        test_target_score_drop0,
        test_loss_drop0,
        test_base_acc1_drop0,
        test_base_acc5_drop0,
        test_targets_drop0
    ) = join_list_of_tuples(predict_results[1])

    quantile_dropped = private_predicted_quantile_threshold_drop0[:, -4]
    quantile_kept = private_predicted_quantile_threshold[:, -4]
    quantile_dropped = quantile_dropped[private_targets_drop0 == 0]
    quantile_kept = quantile_kept[private_targets == 0]

    # Plot the distributions of quantile thresholds for the two classes
    plt.figure(figsize=(15, 10))
    plt.hist(quantile_dropped, bins=50, alpha=0.2, color='blue', label='Quantile Dropped')
    plt.hist(quantile_kept, bins=50, alpha=0.2, color='orange', label='Quantile Kept')
    plt.xlabel('Quantile Threshold Values')
    plt.ylabel('Density')
    plt.title('Distribution of Quantile Threshold Values for Class 0')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('qmia_quantile_dropped_vs_kept.png')
    plt.close()

    # Plot the difference between threshold predicted with sample in training vs not included in training
    quantile_difference = quantile_dropped - quantile_kept
    plt.figure(figsize=(15, 10))
    plt.hist(quantile_difference, bins=50, alpha=0.7, color='green', label='Quantile Difference')
    plt.xlabel('Quantile Difference Values')
    plt.ylabel('Density')
    plt.title('Difference in Quantile Threshold Values for Class 0')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('qmia_quantile_difference.png')
    plt.close()

def compare_id_ood(args):
    
    fig_name="best"
    prediction_output_dir = 'models/imagenet-1k_cifar100/0_16_0_16/mia/gaussian_qmia/facebook/convnext-large-224-22k-1k/use_hinge_True/use_target_label_False/use_target_inputs_False/cls_drop_/predictions/best'

    predict_results = None
    for file in glob(os.path.join(prediction_output_dir, "*.pt")):
        print(file)
        rank_predict_results = torch.load(file)
        if predict_results is None:
            predict_results = rank_predict_results
        else:
            for r, p in zip(rank_predict_results, predict_results):
                p.extend(r)

    def join_list_of_tuples(list_of_tuples):
        n_tuples = len(list_of_tuples[0])
        result = []
        for _ in range(n_tuples):
            try:
                result.append(torch.concat([t[_] for t in list_of_tuples]))
            except:
                result.append(torch.Tensor([t[_] for t in list_of_tuples]))
        return result

    (
        private_predicted_quantile_threshold,
        private_target_score,
        private_loss,
        private_base_acc1,
        private_base_acc5,
        private_targets,
        private_p_values,
    ) = join_list_of_tuples(predict_results[-1])
    (
        test_predicted_quantile_threshold,
        test_target_score,
        test_loss,
        test_base_acc1,
        test_base_acc5,
        test_targets,
        test_p_values,
    ) = join_list_of_tuples(predict_results[1])

    private_mean = private_predicted_quantile_threshold[:, 0]

    # Plot the distributions of quantile thresholds against the target score
    plt.figure(figsize=(15, 10))
    plt.scatter(private_mean, private_target_score, alpha=0.2, color='blue', label='Predicted Mean vs Target Score')
    plt.xlabel('Predicted Mean Score')
    plt.ylabel('Target Score')
    plt.title('Distribution of Predicted Mean vs Target Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('qmia_mean_vs_target_score.png')
    plt.close()

def get_cifar100_predictions_from_inet_mia_model():
    import os
    import shutil
    import warnings

    warnings.simplefilter("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

    import math
    from glob import glob
    import time

    import numpy as np
    import pytorch_lightning as pl
    import torch
    from torch import distributed as dist
    from analysis_utils import plot_performance_curves, plot_performance_curves_id_ood
    from data_utils import CustomDataModule
    from lightning_utils import CustomWriter, LightningQMIA
    from train_mia_ray import argparser
    from tqdm import tqdm

    datamodule = CustomDataModule(
                dataset_name="cifar100/0_16",
                mode="eval",
                num_workers=7,
                image_size=32,
                batch_size=16,
                data_root="./data/",
            )

    dir = "./models/imagenet-1k_cifar100/0_16_0_16/mia/gaussian_qmia/facebook/convnext-large-224-22k-1k/use_hinge_True/use_target_label_False/use_target_inputs_False/cls_drop_/"
    lightning_model = LightningQMIA.load_from_checkpoint(os.path.join(dir, "best_val_loss.ckpt"))

    prediction_output_dir = os.path.join(
                dir,
                "predictions",
                "cifar100",
            )
    pred_writer = CustomWriter(
                output_dir=prediction_output_dir,
                write_interval="epoch",
            )
    trainer = pl.Trainer(
                max_epochs=1,
                accelerator="auto" if torch.cuda.is_available() else "cpu",
                callbacks=[pred_writer],
                devices=-1 if torch.cuda.is_available() else 1,
                enable_progress_bar=True,
            )
    trainer.predict(
                lightning_model,
                datamodule,
                return_predictions=False,
            )

if __name__ == "__main__":
    args = argparser()
    get_cifar100_predictions_from_inet_mia_model()