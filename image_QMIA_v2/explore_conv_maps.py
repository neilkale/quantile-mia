import os
import argparse
import random
from lightning_utils import LightningQMIA, CustomWriter
from data_utils import CustomDataModule
import pytorch_lightning as pl

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

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

def visualize_mia_conv_maps(args, model, input_image_path=None, layer_names=None):
    """
    Visualize convolutional feature maps from any CNN model with Conv2d layers.
    
    Args:
        args: Arguments containing paths for saving results
        model: The model to visualize (already loaded and in eval mode)
        input_image_path: Path to input image (if None, will use random noise)
        layer_names: List of specific layer names to visualize (if None, will visualize all Conv2d layers)
    """
    # Create save directory if it doesn't exist
    save_dir = os.path.join(args.attack_plots_path, "conv_maps")
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare input data
    if input_image_path:
        # Load and preprocess a real image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_image = Image.open(input_image_path).convert('RGB')
        input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
    else:
        # Generate random noise as input
        input_tensor = torch.randn(1, 3, 224, 224)
    
    # Move to the same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Dictionary to store feature maps
    feature_maps = {}
    
    # Function to get conv layers and their names
    def get_conv_layers(model, prefix=""):
        layers = []
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # If it's a Conv2d layer, add it to our list
            if isinstance(module, torch.nn.Conv2d):
                layers.append((full_name, module))
            
            # If it's a container (Sequential, BasicBlock, etc.), recursively get its conv layers
            if len(list(module.children())) > 0:
                layers.extend(get_conv_layers(module, full_name))
                
        return layers
    
    # Get all conv layers or filter based on provided layer_names
    all_conv_layers = get_conv_layers(model)
    if layer_names:
        conv_layers = [(name, module) for name, module in all_conv_layers if name in layer_names]
    else:
        conv_layers = all_conv_layers
    
    # Register hooks to capture feature maps
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook
    
    for name, layer in conv_layers:
        hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Additional hooks for capturing intermediate representations 
    # This needs to be model-agnostic
    final_features = {}
    
    def capture_intermediate_features():
        """
        Identify and capture key intermediate features without assuming a specific architecture.
        Uses heuristics to find pooling and final features.
        """
        # Find pooling layers
        pooling_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d)):
                pooling_layers.append((name, module))
        
        # Register hooks for pooling layers
        for name, layer in pooling_layers:
            def hook_fn_pool(pool_name):
                def hook(module, input, output):
                    final_features[f"{pool_name}_output"] = output.detach().cpu()
                return hook
            hooks.append(layer.register_forward_hook(hook_fn_pool(name)))
        
        # Find final convolutional layer
        conv_layer_names = [name for name, _ in conv_layers]
        if conv_layer_names:
            # Sort by depth (assuming deeper layers have longer names due to nesting)
            sorted_conv_layers = sorted(conv_layer_names, key=lambda x: len(x.split('.')), reverse=True)
            final_conv_names = [name for name in sorted_conv_layers if len(name.split('.')) == max(len(n.split('.')) for n in conv_layer_names)]
            
            # Register hooks for final convolutional layers
            for name, layer in conv_layers:
                if name in final_conv_names:
                    def hook_fn_final(final_name):
                        def hook(module, input, output):
                            # Store output directly
                            final_features[f"{final_name}_final"] = output.detach().cpu()
                            
                            # Try common pooling operations to capture what might happen after
                            try:
                                # Adaptive pooling to 1x1 (common in many networks)
                                pooled = F.adaptive_avg_pool2d(output, 1)
                                final_features[f"{final_name}_adaptive_pooled"] = pooled.detach().cpu()
                                
                                # Flattened features
                                final_features[f"{final_name}_flattened"] = pooled.view(pooled.size(0), -1).detach().cpu()
                            except:
                                pass
                        return hook
                    hooks.append(layer.register_forward_hook(hook_fn_final(name)))
    
    # Set up the intermediate feature capture
    capture_intermediate_features()
    
    try:
        # Forward pass to get activations
        with torch.no_grad():
            # Simple forward pass - works with any model
            output = model(input_tensor)
    except Exception as e:
        print(f"Error during forward pass: {e}")
        print("Attempting to capture feature maps despite the error...")
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Merge the feature maps
    feature_maps.update(final_features)
    
    # Print summary of captured feature maps
    print(f"Captured {len(feature_maps)} feature maps:")
    for name, feat_map in feature_maps.items():
        print(f"  {name}: {feat_map.shape}")
    
    # Visualize the feature maps
    for name, feature_map in feature_maps.items():
        # Skip non-visual feature maps (like flattened vectors)
        if len(feature_map.shape) < 4 or "flattened" in name:
            print(f"Skipping visualization for {name}: {feature_map.shape}")
            continue
        visualize_feature_map(name, feature_map, save_dir)
    
    return feature_maps

def visualize_feature_map(name, feature_map, save_dir):
    """Visualize a feature map from a specific layer."""
    # Take first item in batch
    feature_map = feature_map[0]  # Shape: [channels, height, width]
    
    # Check if this is not a proper feature map (e.g., it's flattened or 1D)
    if len(feature_map.shape) < 3:
        print(f"Skipping visualization for {name} as it's not a 2D feature map: {feature_map.shape}")
        return
    
    # Determine number of channels to display
    num_channels = feature_map.shape[0]
    max_channels = min(64, num_channels)  # Limit to reasonable number
    
    # Calculate grid size - try to make it roughly square
    grid_size = int(np.ceil(np.sqrt(max_channels)))
    
    # Create figure
    plt.figure(figsize=(20, 20))
    plt.suptitle(f"Feature Maps: {name} (showing {max_channels}/{num_channels} channels)", fontsize=16)
    
    # Plot each channel
    for i in range(max_channels):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Get channel data and normalize for better visualization
        channel_data = feature_map[i].numpy()
        vmin, vmax = channel_data.min(), channel_data.max()
        if vmax > vmin:  # Avoid division by zero
            channel_data = (channel_data - vmin) / (vmax - vmin)
        
        plt.imshow(channel_data, cmap='viridis')
        plt.axis('off')
        plt.title(f'Ch {i}')
    
    # Save figure
    safe_name = name.replace('.', '_').replace('/', '_')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"feature_map_{safe_name}.png"), dpi=150)
    plt.close()

def visualize_conv_filters(args, model, layer_names=None):
    """
    Visualize the convolutional filters (weights) of any CNN model.
    
    Args:
        args: Arguments containing paths for saving results
        model: The model to visualize (already loaded)
        layer_names: List of specific layer names to visualize (if None, will visualize all Conv2d layers)
    """
    # Create save directory if it doesn't exist
    save_dir = os.path.join(args.attack_plots_path, "filters")
    os.makedirs(save_dir, exist_ok=True)
    
    # Function to get conv layers and their names
    def get_conv_layers(model, prefix=""):
        layers = []
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # If it's a Conv2d layer, add it to our list
            if isinstance(module, torch.nn.Conv2d):
                layers.append((full_name, module))
            
            # If it's a container, recursively get its conv layers
            if len(list(module.children())) > 0:
                layers.extend(get_conv_layers(module, full_name))
                
        return layers
    
    # Get all conv layers or filter based on provided layer_names
    all_conv_layers = get_conv_layers(model)
    if layer_names:
        conv_layers = [(name, module) for name, module in all_conv_layers if name in layer_names]
    else:
        conv_layers = all_conv_layers
    
    # Visualize filters for each layer
    for name, layer in conv_layers:
        # Get weights - shape is [out_channels, in_channels, kernel_height, kernel_width]
        weights = layer.weight.detach().cpu()
        
        # Print shape information
        print(f"Layer {name} weights shape: {weights.shape}")
        
        # Visualize the filters
        visualize_filter(name, weights, save_dir)

def visualize_filter(name, weights, save_dir):
    """
    Visualize convolutional filters.
    
    Args:
        name: Layer name
        weights: Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width]
        save_dir: Directory to save visualizations
    """
    out_channels, in_channels, k_height, k_width = weights.shape
    
    # Skip tiny filters (upsampling them is not very informative)
    if k_height < 2 and k_width < 2:
        print(f"Skipping visualization for {name} as filters are 1x1: {weights.shape}")
        return
    
    # Limit to a reasonable number of output channels
    max_out_channels = min(64, out_channels)
    
    # For the first convolutional layer (typically processing RGB input), we can visualize in color
    if in_channels == 3 and ('conv1' in name.lower() or 'stem' in name.lower() or 'downsample' in name.lower()):
        plt.figure(figsize=(20, 20))
        plt.suptitle(f"RGB Filters: {name} (showing {max_out_channels}/{out_channels} filters)", fontsize=16)
        
        grid_size = int(np.ceil(np.sqrt(max_out_channels)))
        
        for i in range(max_out_channels):
            plt.subplot(grid_size, grid_size, i + 1)
            
            # For tiny filters, we'll upsample for better visibility
            if k_height < 5 or k_width < 5:
                # Upsample using bicubic interpolation
                kernel = weights[i].permute(1, 2, 0).numpy()
                kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
                
                pil_kernel = Image.fromarray((kernel * 255).astype(np.uint8))
                upsampled = pil_kernel.resize((64, 64), Image.BICUBIC)
                kernel = np.array(upsampled) / 255.0
            else:
                # Transpose to [height, width, channels] for RGB display
                kernel = weights[i].permute(1, 2, 0).numpy()
                kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
            
            plt.imshow(kernel)
            plt.axis('off')
            plt.title(f'Filter {i}')
        
        safe_name = name.replace('.', '_').replace('/', '_')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"filter_rgb_{safe_name}.png"), dpi=150)
        plt.close()
    
    # For all layers, visualize each output channel's filters
    # (For deeper layers with many input channels, we'll visualize a subset)
    max_in_channels = min(9, in_channels)  # Limit input channels to visualize
    
    for out_idx in range(max_out_channels):
        plt.figure(figsize=(15, 15))
        plt.suptitle(f"Filters: {name}, Output Ch {out_idx} (showing {max_in_channels}/{in_channels} input chs)", fontsize=16)
        
        grid_size = int(np.ceil(np.sqrt(max_in_channels)))
        
        for in_idx in range(max_in_channels):
            plt.subplot(grid_size, grid_size, in_idx + 1)
            
            # Get the 2D kernel for this in/out channel combination
            kernel = weights[out_idx, in_idx].numpy()
            
            # For tiny filters, we'll upsample for better visibility
            if k_height < 5 or k_width < 5:
                # Normalize to 0-255 range for PIL
                kernel_norm = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8) * 255
                pil_kernel = Image.fromarray(kernel_norm.astype(np.uint8), 'L')
                upsampled = pil_kernel.resize((64, 64), Image.BICUBIC)
                kernel = np.array(upsampled) / 255.0
            else:
                # Normalize for better visualization
                kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
            
            plt.imshow(kernel, cmap='viridis')
            plt.axis('off')
            plt.title(f'In Ch {in_idx}')
        
        safe_name = name.replace('.', '_').replace('/', '_')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"filter_{safe_name}_out{out_idx}.png"), dpi=150)
        plt.close()

def visualize_cam(args, model, input_tensor, target_class=None):
    """
    Compute and visualize Class Activation Maps.
    
    Args:
        args: Arguments containing paths for saving results
        model: The model (already loaded and in eval mode)
        input_tensor: Input tensor of shape [1, 3, H, W]
        target_class: Target class index (if None, will use predicted class)
    """
    # Create save directory
    save_dir = os.path.join(args.attack_plots_path, "cam")
    os.makedirs(save_dir, exist_ok=True)
    
    # Move input to the same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Get the last convolutional layer
    last_conv_layer = None
    
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            last_conv_layer = (name, module)
            break
    
    if last_conv_layer is None:
        print("Could not find a convolutional layer in the model.")
        return
    
    last_conv_name, last_conv_module = last_conv_layer
    
    # Register hook to get activations from last conv layer
    activations = None
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    handle = last_conv_module.register_forward_hook(hook_fn)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        
        # If target class is not provided, use predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
    
    # Remove hook
    handle.remove()
    
    # Find the weights of the final fully connected layer
    fc_weights = None
    
    # Try to find the final fully connected layer or equivalent
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Linear):
            # Check if this is the classification layer
            if module.out_features > 1:  # Assuming multi-class classification
                fc_weights = module.weight[target_class].detach().cpu()
                break
    
    if fc_weights is None:
        print("Could not find fully connected layer weights for CAM.")
        return
    
    # Reshape weights if needed
    if fc_weights.shape[0] != activations.shape[1]:
        # This might happen in architectures with global pooling or different connectivity
        print(f"Warning: FC weights shape {fc_weights.shape} doesn't match activations channels {activations.shape[1]}.")
        print("Attempting to compute CAM anyway, but results may not be accurate.")
        
        # Try to adapt weights if possible
        if hasattr(model, 'avgpool') and isinstance(model.avgpool, torch.nn.AdaptiveAvgPool2d):
            # This is a common case in many CNNs
            fc_weights = fc_weights.view(-1, activations.shape[1])
        else:
            # As a fallback, just use average pooling on activations
            pooled_activations = F.adaptive_avg_pool2d(activations, (1, 1))
            pooled_activations = pooled_activations.view(pooled_activations.size(0), -1)
            
            # Compute importance weights based on gradient
            # This is a simple approximation when the exact architectural details are unknown
            dummy_output = torch.sum(pooled_activations * fc_weights)
            dummy_output.backward(retain_graph=True)
            weights = pooled_activations.grad[0].cpu()
    else:
        weights = fc_weights
    
    # Get the activations
    act = activations.cpu().numpy()[0]  # Remove batch dimension
    
    # Compute weighted sum of activation maps
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    
    # Apply ReLU to focus on features that have a positive influence
    cam = np.maximum(cam, 0)
    
    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Resize to match input size
    from skimage.transform import resize
    cam = resize(cam, (input_tensor.shape[2], input_tensor.shape[3]), preserve_range=True)
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    img = input_tensor[0].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # CAM
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title(f'Class Activation Map (Class {target_class})')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cam_class{target_class}.png"), dpi=150)
    plt.close()
    
    return cam

if __name__ == "__main__":
    args = argparser()

    print("Visualizing MIA Conv Maps...")
    lightning_model = LightningQMIA.load_from_checkpoint(os.path.join(
        args.attack_checkpoint_path,
        "best_val_loss.ckpt"
    ))
    lightning_model.eval()
    lightning_model.freeze()
    lightning_model.model.eval()
    
    visualize_mia_conv_maps(
        args, lightning_model.model,
        input_image_path=os.path.join(args.attack_plots_path, "cifar10-train-156.png"),
        layer_names=None
    )
    