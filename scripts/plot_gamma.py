import yaml
import torch
import os
import sys
import torch_pruning as tp
import torch.nn as nn
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    
    # if isinstance(checkpoint, dict) and "model" in checkpoint:
    #     state_dict = checkpoint["model"].state_dict() 
    # elif isinstance(checkpoint, Model):  # Whole model is stored
    #     state_dict = checkpoint.state_dict()  
    # else:
    #     raise ValueError("Checkpoint does not contain a valid YOLOv5 model or state_dict.")

    # Initialize model
    # model = Model(cfg_path).to(device)
    model = Model(cfg_path or checkpoint['model'].yaml, ch=3, nc=1).to(device)  # create
    # Load weights
    # model.load_state_dict(state_dict)
    
    
    # Evaluation mode
    model.eval()

    # Return number of params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    
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

def plot_gamma_distribution_3d(model, save_path="gamma_distribution_baseline2.png"):
    """
    Plots the 3D distribution of BatchNorm γ values.
    X-axis: γ values
    Y-axis: Layer index
    Z-axis: Frequency
    """
    gamma_values = []
    layer_indices = []

    # Zbierame γ hodnoty z BatchNorm vrstiev
    for i, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.BatchNorm2d):
            gamma_values.append(m.weight.data.abs().cpu().numpy())
            layer_indices.extend([i] * len(m.weight.data))

    # Konverzia do numpy array
    gamma_values = np.concatenate(gamma_values)
    layer_indices = np.array(layer_indices)

    # 2D histogram (γ hodnoty vs. vrstvy)
    hist, xedges, yedges = np.histogram2d(gamma_values, layer_indices, bins=(50, len(set(layer_indices))))

    # Príprava gridu
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    # Výšky (frekvencie γ hodnôt)
    dz = hist.ravel()

    # 3D plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, 0.1, 1, dz, shade=True)

    ax.set_xlabel("BN γ values")
    ax.set_ylabel("Layer index")
    ax.set_zlabel("Frequency")
    ax.set_title("3D Distribution of BatchNorm γ values")

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
        # 'anchors': anchors
        # # Anchor boxes
    }
    
    
    save_path2 = "yolov5s_pruned.yaml"
    # Ak model má aktualizovanú konfiguráciu v atribúte 'yaml'
    if hasattr(model, 'yaml'):
        cfg_dict = model.yaml
        with open(save_path2, "w") as f:
            yaml.dump(cfg_dict, f)
        print(f"Pruned model config saved to {save_path2}")
    else:
        print("Model does not have a YAML configuration attribute. Budete musieť vytvoriť konfiguráciu manuálne.")




    # Save with compression enabled to minimize size
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
    # torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)  # Kompresia
    print(f"Model uložený do {save_path} (veľkosť: {os.path.getsize(save_path)/1e6:.2f} MB)")

    # # Save the model checkpoint
    # with torch.no_grad():
    #     torch.save(checkpoint, save_path)
    # print(f"Pruned model saved to {save_path}")

def print_model_channel_info(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"{name} - Conv weight shape: {module.weight.shape}")
        elif isinstance(module, torch.nn.BatchNorm2d):
            print(f"{name} - BN weight shape: {module.weight.shape}")






if __name__ == "__main__":
    # File paths
    # cfg_path = "yolov5/models/yolov5s_custom.yaml"
    cfg_path = "yolov5/models/yolov5s-BIFPN.yaml"
    # weights_path = "../models/24hour/exp48/weights/best.pt" # eca layer
    # weights_path = "../models/500epochs/weights/best.pt"
    # weights_path = "../models/eca/weights/last.pt"
    # weights_path = "../models/eca-0.5/weights/best.pt"
    # weights_path = "../models/exp28/weights/best.pt"
    weights_path = "../models/sparsity0.0005/weights/best.pt"
    # weights_path = "yolov5/runs/train/exp15/weights/best.pt"
    # weights_path = "baseline.pt"
    # weights_path = "pruned_model_baseline.pt"
    # weights_path = "sparsity_pruned2.pt"
    save_path = "last_pruned_model.pt"
    # weights_path = "last_pruned_model.pt"

    # Load model
    original_model = load_model(cfg_path, weights_path)
    print("Model loaded successfully!")


    # Example input tensor
    # example_input = torch.randn(1, 3, 640, 640)

    # # Plot gamma distribution
    plot_gamma_distribution(original_model, save_path="gamma_distribution3.png")
    plot_gamma_distribution_3d(original_model, save_path="gamma_distribution_baseline2.png")
    
    # Apply pruning
    # pruned_model = hard_prune_model(original_model)
    # print("Model pruned successfully!")
    
    # pruned_model = prune_model(original_model, img_tensor)
    # print("Model pred pruningom:")
    # print_model_channel_info(original_model)

    # save_pruned_model(pruned_model, save_path, nc=1, names=['QR'], anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]])
    # print("\nModel po pruning-u:")
    # print_model_channel_info(pruned_model)

    # Save pruned model
