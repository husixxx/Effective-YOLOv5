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
YOLOV5_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5_custom'))


if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)
    
from models.yolo import Model, Detect


def hard_prune_conv_bn_block(conv_module, bn_module, prune_ratio=0.9):
    """
    Fyzicky odstráni kanály v Conv+BN bloku, kde BN váhy (gamma) patria medzi najnižších podľa prune_ratio.
    
    Args:
        conv_module: Konvolučná vrstva (nn.Conv2d).
        bn_module: Prislúchajúca BatchNorm vrstva (nn.BatchNorm2d).
        prune_ratio: Frakcia kanálov, ktoré sa majú odstrániť (napr. 0.9 = odstrániť 90% najnižších).
        
    Returns:
        new_conv: Upravená konvolučná vrstva s menším počtom výstupných kanálov.
        new_bn: Upravená BatchNorm vrstva s menším počtom kanálov.
        keep_idx: Indexy kanálov, ktoré boli ponechané.
    """
    
    gamma = bn_module.weight.data.abs()
    total_channels = gamma.numel()
    num_to_prune = int(total_channels * prune_ratio)
    
    
    sorted_idx = torch.argsort(gamma)
    prune_idx = sorted_idx[:num_to_prune]
    
    
    keep_idx = [i for i in range(total_channels) if i not in prune_idx.tolist()]
    
    if len(keep_idx) == 0:
        raise ValueError("No channels left after pruning!")
    
   
    new_conv_weight = conv_module.weight.data[keep_idx, :, :, :].clone()
    in_channels = conv_module.in_channels
    kernel_size = conv_module.kernel_size
    stride = conv_module.stride
    padding = conv_module.padding
    new_out_channels = len(keep_idx)
    
    
    new_conv = nn.Conv2d(in_channels, new_out_channels, kernel_size, stride, padding, bias=False)
    new_conv.weight.data = new_conv_weight
    
    # Upravenie BN vrstvy
    new_bn_weight = bn_module.weight.data[keep_idx].clone()
    new_bn_bias = bn_module.bias.data[keep_idx].clone()
    new_bn_running_mean = bn_module.running_mean[keep_idx].clone()
    new_bn_running_var = bn_module.running_var[keep_idx].clone()
    new_bn = nn.BatchNorm2d(new_out_channels)
    new_bn.weight.data = new_bn_weight
    new_bn.bias.data = new_bn_bias
    new_bn.running_mean = new_bn_running_mean
    new_bn.running_var = new_bn_running_var
    
    return new_conv, new_bn, keep_idx


def hard_prune_model(model, prune_ratio=0.2):
    """
    Aplikuje hard pruning na všetky Conv+BN bloky v modeli.
    UPOZORNENIE: Táto funkcia upravuje iba bloky, ktoré majú atribúty 'conv' a 'bn'. 
    Pre úplnú rekonštrukciu modelu je potrebné aktualizovať aj následné vrstvy, 
    ktoré prijímajú výstup z týchto blokov.
    
    Args:
        model: YOLOv5 model (napr. model z yolov5s.yaml).
        prune_ratio: Frakcia kanálov, ktoré sa odstránia v každom BN bloku.
    
    Returns:
        Upravený model.
    """

    for name, module in model.named_children():
        
        if hasattr(module, 'conv') and hasattr(module, 'bn'):
           
            new_conv, new_bn, keep_idx = hard_prune_conv_bn_block(module.conv, module.bn, prune_ratio)
           
            module.conv = new_conv
            module.bn = new_bn
            print(f"Pruned block {name}: kept {len(keep_idx)}/{module.bn.weight.data.numel()+len(keep_idx)} channels.")
        else:
            
            hard_prune_model(module, prune_ratio)
    return model


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


def prune_bn_by_gamma(model, example_input, prune_ratio=0.2, iterative_steps=1):
    """
    Prune model's BatchNorm layers based on gamma values.

    Args:
        model: YOLOv5 model.
        example_input: Input tensor for DependencyGraph creation.
        prune_ratio: Ratio of BN channels to remove.
        iterative_steps: Number of pruning iterations.

    Returns:
        Pruned model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    example_input = example_input.to(device)

    
    importance_criterion = tp.importance.MagnitudeImportance(p=1)

    
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)

    
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            gamma_values = m.weight.data.abs().cpu().numpy()  # Extract BN gamma values
            
           
            sorted_idx = gamma_values.argsort()

            
            num_channels_to_prune = int(len(sorted_idx) * prune_ratio)

           
            prune_idx = sorted_idx[:num_channels_to_prune]

            pruning_plan = DG.get_pruning_plan(m, tp.prune_batchnorm, idxs=prune_idx)
            pruning_plan.exec()  # Execute pruning

    # Print final stats
    final_macs, final_params = tp.utils.count_ops_and_params(model, example_input)
    print(f"After pruning: MACs={final_macs}, Params={final_params}")

    return model

def prune_model_r1(model, prune_ratio=0.9):
    # Step 1: Collect all BN γ values and their corresponding Conv layers
    bn_conv_pairs = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            # Find the preceding Conv layer (assuming naming convention like "model.0.conv" -> "model.0.bn")
            conv_name = name.replace(".bn", ".conv")
            conv_module = None
            for n, m in model.named_modules():
                if n == conv_name and isinstance(m, torch.nn.Conv2d):
                    conv_module = m
                    break
            if conv_module is not None:
                bn_conv_pairs.append((module, conv_module))

    # Step 2: Sort all BN γ values globally
    all_gammas = torch.cat([bn.weight.data.abs().flatten() for bn, _ in bn_conv_pairs])
    sorted_gammas, _ = torch.sort(all_gammas)
    threshold_idx = int(len(sorted_gammas) * prune_ratio)
    threshold = sorted_gammas[threshold_idx]

    # Step 3: Prune channels where γ < threshold
    for bn, conv in bn_conv_pairs:
        mask = bn.weight.data.abs().ge(threshold).float().to(bn.weight.device)
        
        # Prune BN layer
        bn.weight.data.mul_(mask)
        bn.bias.data.mul_(mask)
        
        # Prune Conv layer (output channels)
        # Conv weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        conv.weight.data.mul_(mask.view(-1, 1, 1, 1))  # Correct dimension: [out_channels, 1, 1, 1]

    return model

def prune_model_r1_correct(model, prune_ratio=0.2):
    """
    Prunes the model based on the BN gamma values in the Conv blocks,
    following the algorithm in the paper:
      1. Collect all BN gamma values from Conv modules.
      2. Sort them and determine a threshold based on prune_ratio.
      3. For each Conv block (which has 'conv' and 'bn' submodules), 
         prune channels where BN gamma < threshold.
    
    Args:
        model: YOLOv5 model to prune.
        prune_ratio: Fraction of channels (based on sorted gamma values) to prune.
        
    Returns:
        Pruned model.
    """
    
    bn_list = []
    conv_bn_modules = []
    for m in model.modules():
        
        if hasattr(m, 'bn') and hasattr(m, 'conv'):
            bn_list.append(m.bn.weight.data.flatten())
            conv_bn_modules.append(m)
    
    if len(bn_list) == 0:
        print("No BN layers found in the model!")
        return model

   
    all_gammas = torch.cat(bn_list)
    sorted_gammas, _ = torch.sort(all_gammas)
    
    
    threshold = sorted_gammas[int(len(sorted_gammas) * prune_ratio)]
    print(f"Pruning threshold (gamma): {threshold}")

    
    for module in conv_bn_modules:
        gamma = module.bn.weight.data
       
        mask = gamma.abs().ge(threshold).float().to(gamma.device)
       
        module.bn.weight.data.mul_(mask)
        module.bn.bias.data.mul_(mask)
       
        module.conv.weight.data.mul_(mask.view(-1, 1, 1, 1))
    
   
    for param in model.parameters():
        param.requires_grad = False

    return model


def prune_model(model, example_input, iterative_steps=5, backbone_sparsity=0.5, head_sparsity=0.2):
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


    importance_criterion = tp.importance.MagnitudeImportance(p=1)  # Or tp.importance.MagnitudeImportance()

   
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, Detect):
            ignored_layers.append(m)
        # if hasattr(m, 'bn') and hasattr(m, 'conv'):
        #     print("Skipping head layer:", m)
        # else:
        #     ignored_layers.append(m)

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
    cfg_path = "yolov5_custom/models/yolov5s.yaml"
    # weights_path = "../models/24hour/exp48/weights/best.pt" # eca layer
    # weights_path = "../models/500epochs/weights/best.pt"
    # weights_path = "../models/eca/weights/last.pt"
    weights_path = "../models/eca-0.5/weights/best.pt"
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

    # Plot gamma distribution
    plot_gamma_distribution(original_model, save_path="gamma_distribution3.png")
    plot_gamma_distribution_3d(original_model, save_path="gamma_distribution_baseline2.png")
    
    # Apply pruning
    # pruned_model = hard_prune_model(original_model)
    # print("Model pruned successfully!")
    
    pruned_model = prune_model(original_model, img_tensor)
    # print("Model pred pruningom:")
    # print_model_channel_info(original_model)

    save_pruned_model(pruned_model, save_path, nc=1, names=['QR'], anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]])
    # print("\nModel po pruning-u:")
    # print_model_channel_info(pruned_model)

    # Save pruned model
