import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from torch_models import get_cifar_resnet_model, get_torchvision_model

from lightning_utils import LightningQMIA, CustomWriter

from tqdm import tqdm
from data_utils import CustomDataModule
import pytorch_lightning as pl
import glob

# Configuration
MODEL_ARCH = "convnext-tiny"  # or "resnet50"
CHECKPOINT_PATH = "/work3/nkale/ml-projects/classification-quantile-mia/image_QMIA_v2/models/mia/base_cinic10/0_16/cifar-resnet-18/attack_cinic10/0_16/facebook/convnext-tiny-224/score_fn_top_two_margin/loss_fn_gaussian/cls_drop_0/best_val_loss.ckpt"
BATCH_SIZE = 32
NUM_CLASSES = 10
CINIC_TEST_DIR = "/work3/nkale/ml-projects/classification-quantile-mia/image_QMIA_v3/data/cinic10/test"  # Update with your CINIC-10 test directory path

# Define the CINIC-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Custom Dataset for CINIC-10
class CINIC10Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# Set up data transforms
transform = transforms.Compose([
#    transforms.Resize(224),
#    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the model
def load_model(architecture, checkpoint_path):
    if architecture == "resnet18":
        model = get_cifar_resnet_model("cifar-resnet-18")
    elif architecture == "resnet50":
        model = get_cifar_resnet_model("cifar-resnet-50")
    elif architecture == 'convnext-tiny':
        model = get_torchvision_model("convnext-tiny")

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        lightning_model = LightningQMIA.load_from_checkpoint(os.path.join(
            checkpoint_path
        ))
        lightning_model.eval()
    
    return model

# Initialize the dataset and dataloader
def get_test_dataloader():
    try:
        test_dataset = CINIC10Dataset(root_dir=CINIC_TEST_DIR, transform=transform)
    except Exception as e:
        print(f"Error loading CINIC-10 dataset: {e}")
        raise
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return test_loader

# Function to run inference and collect logit differences
def run_inference(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Initialize lists to store results for each class
    class_scores = [[] for _ in range(NUM_CLASSES)]
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            print(outputs.shape)
            print(outputs[0])
            
            # For each example, find the difference between top two logits
            for i, (output, label) in enumerate(zip(outputs, labels)):
                # Store the difference in the appropriate class list
                class_scores[label.item()].append(output)
    
    return class_scores

# Function to plot histograms
def plot_logit_diff_histograms(class_logit_diffs):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()
    
    for i, diffs in enumerate(class_logit_diffs):
        if len(diffs) == 0:
            print(f"Warning: No samples for class {class_names[i]}")
            continue

        axs[i].hist(diffs, bins=30, alpha=0.7)
        axs[i].set_title(f'Class: {class_names[i]}')
        axs[i].set_xlabel('Predicted Mean')
        axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('threshold_prediction_histograms.png')
    plt.show()

def main():
    # Load model
    model = load_model(MODEL_ARCH, CHECKPOINT_PATH)
    
    # Get test dataloader
    test_loader = get_test_dataloader()
    
    # Run inference and collect logit differences
    print("Running inference...")
    class_logit_diffs = run_inference(model, test_loader)
    
    # Plot histograms
    print("Plotting histograms...")
    plot_logit_diff_histograms(class_logit_diffs)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for i, diffs in enumerate(class_logit_diffs):
        if len(diffs) > 0:
            print(f"Class {class_names[i]}: Mean={np.mean(diffs):.4f}, Std={np.std(diffs):.4f}, Min={np.min(diffs):.4f}, Max={np.max(diffs):.4f}")

if __name__ == "__main__":
    main()
