import torch
import os
import sys
import torch_pruning as tp
import torch.nn as nn
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
# Load a real image
img = cv2.imread('test1.jpg')
img = cv2.resize(img, (640, 640))  # Resizing to model input size
img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

# 5 559 906
YOLOV5_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5'))


if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)
    
from models.yolo import Model, Detect




def load_model(cfg_path, weights_path):
    """
    Load model.
    Args:
        cfg_path: Yaml file with architecture.
        weights_path: Model weights.
    Returns:
        model: YOLOv5 model.
    """
    device = "cuda"

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"].state_dict() 
    elif isinstance(checkpoint, Model):  # Whole model is stored
        state_dict = checkpoint.state_dict()  
    else:
        raise ValueError("Checkpoint does not contain a valid YOLOv5 model or state_dict.")

    # Initialize model
    model = Model(cfg_path).to(device)

    # Load weights
    model.load_state_dict(state_dict)
    
    
    # Evaluation mode
    model.eval()

    # Return number of params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    
    return model




def prune_model(model, example_input, iterative_steps=1, backbone_sparsity=0.1, head_sparsity=0.2):
    """
    Perform structured channel pruning using MagnitudePruner and iterative steps.

    Args:
        model: YOLOv5 model to prune.
        example_input: Input tensor for DependencyGraph creation.
        iterative_steps: Number of pruning iterations.
        backbone_sparsity: Sparsity level for backbone layers.
        head_sparsity: Sparsity level for head layers.
    Returns:
        Pruned model.
    """
    device = "cuda"
    model = model.to(device)
    example_input = example_input.to(device)

    # Define importance criterion (e.g., Taylor or Magnitude)
    importance_criterion = tp.importance.MagnitudeImportance(p=1)  # Or tp.importance.MagnitudeImportance()

    # Ignore final layers (e.g., YOLO head layers) during pruning
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, Detect):  # Explicitne ignorujeme detekčné vrstvy
            ignored_layers.append(m)

    # Initialize pruner
    pruner = tp.pruner.MagnitudePruner(
        model=model,
        example_inputs=example_input,
        importance=importance_criterion,
        ch_sparsity=backbone_sparsity,
        ignored_layers=ignored_layers,
        iterative_steps=iterative_steps,
    )

    # Print initial stats
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_input)
    print(f"Before pruning: MACs={base_macs}, Params={base_params}")

    # Perform iterative pruning
    for step in range(iterative_steps):

        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, example_input)
        print(f"Step {step + 1}/{iterative_steps}: MACs={macs}, Params={params}")

    for param in model.parameters():
        param.requires_grad = False
    # Final stats
    final_macs, final_params = tp.utils.count_ops_and_params(model, example_input)
    print(f"After pruning: MACs={final_macs}, Params={final_params}")

    return model




def plot_gamma_distribution(model, save_path="gamma_distribution_baseline.png"):
    """
    Plots the distribution of BatchNorm γ values before pruning and saves it to a file.
    """
    gamma_values = []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            gamma_values.append(m.weight.data.abs().cpu().numpy())

    if not gamma_values:
        print("No BatchNorm layers found in the model!")
        return

    gamma_values = np.concatenate(gamma_values)
    plt.hist(gamma_values, bins=50)
    plt.xlabel("BN γ values")
    plt.ylabel("Frequency")
    plt.title("Distribution of BatchNorm γ values before pruning")

    # Uložení grafu jako obrázek místo plt.show()
    plt.savefig(save_path)
    print(f"Histogram uložen do: {os.path.abspath(save_path)}")



def save_pruned_model(model, save_path, nc=1, names=None, anchors=None):
    """
    Save the pruned model (structure + weights) in YOLOv5-compatible format.

    Args:
        model: The pruned model to save.
        save_path: Path to save the model file.
        nc: Number of classes (default: 1).
        names: List of class names (default: None).
        anchors: Anchor boxes (default: None).
    """
    if names is None:
        names = ['class0']  # Default class name if not provided

    model.nc = nc  # Set number of classes
    # model.names = names  # Set class names
    # model.anchors = anchors  # Set anchor boxes
    # model.strip()
    # Prepare the checkpoint dictionary with all necessary info
    checkpoint = {
        'model': model.half()   # Only the model's weights
        # 'nc': nc,                      # Number of classes
        # 'names': names,                # Class names
        # 'anchors': anchors             # Anchor boxes
    }

    # Save with compression enabled to minimize size
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
    # torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)  # Kompresia
    print(f"Model uložený do {save_path} (veľkosť: {os.path.getsize(save_path)/1e6:.2f} MB)")

    # # Save the model checkpoint
    # with torch.no_grad():
    #     torch.save(checkpoint, save_path)
    # print(f"Pruned model saved to {save_path}")


if __name__ == "__main__":
    # File paths
    cfg_path = "yolov5/models/yolov5s.yaml"
    # weights_path = "../models/24hour/exp48/weights/best.pt" // eca layer
    # weights_path = "../models/500epochs/weights/best.pt"
    # weights_path = "../models/exp5/weights/best.pt"
    # weights_path = "yolov5/runs/train/exp15/weights/best.pt"
    weights_path = "baseline.pt"
    # weights_path = "pruned_model_baseline.pt"
    save_path = "sparsity_pruned.pt"

    # Load model
    model = load_model(cfg_path, weights_path)
    print("Model loaded successfully!")

    # Example input tensor
    # example_input = torch.randn(1, 3, 640, 640)

    # Plot gamma distribution
    # plot_gamma_distribution(model)
    # Apply pruning
    pruned_model = prune_model(model, img_tensor)
    print("Model pruned successfully!")

    # Save pruned model
    save_pruned_model(pruned_model, save_path, nc=1, names=['QR'], anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]])
