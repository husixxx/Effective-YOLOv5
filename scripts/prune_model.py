import torch
import torch.nn as nn
from copy import deepcopy

# Load model
model_path = 'runs/train/exp2/weights/best.pt'
model = torch.load(model_path, map_location='cuda')['model'].float()

# Extract convolutional layers
conv_layers = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
original_weights = {id(module): deepcopy(module.weight.data) for module in conv_layers}

# Parameters for pruning
pruning_rate = 0.2  # 20% threshold for pruning
target_pruning = 0.1  # 10% of filters to keep
current_pruning = 1.0  # Start with no pruning (100%)

iteration = 0

# Iterative pruning
while current_pruning > target_pruning:
    iteration += 1
    print(f"== Iterace {iteration} ==")

    # Collect all weights to determine threshold
    all_weights = torch.cat([module.weight.data.view(-1) for module in conv_layers])
    
    # Calculate pruning threshold based on magnitude
    threshold = torch.quantile(all_weights.abs(), pruning_rate)
    print(f"Threshold pro iteraci {iteration}: {threshold.item()}")

    # Apply pruning mask and remove filters
    with torch.no_grad():
        for i, module in enumerate(conv_layers):
            # Get weights and calculate mask
            weights = module.weight.data
            mask = (weights.abs() >= threshold).float()  # 1 for weights >= threshold, 0 for others

            # Identify which filters to keep
            num_filters_to_keep = int(mask.sum().item())  # Count the number of filters to keep

            # Only keep the filters that are above the threshold
            if num_filters_to_keep > 0:
                # Remove the pruned filters (those corresponding to mask == 0)
                keep_indices = mask.bool().nonzero(as_tuple=True)[0]  # Get indices of non-zero weights
                pruned_weights = weights[keep_indices]

                # Reconstruct the layer with the pruned filters
                in_channels, out_channels, kernel_size_h, kernel_size_w = weights.shape
                new_out_channels = len(keep_indices)  # Update the number of output channels

                # Rebuild the conv layer with the remaining filters
                new_conv_layer = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=new_out_channels,
                    kernel_size=(kernel_size_h, kernel_size_w),
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=(module.bias is not None)
                )

                # Update the pruned layer's weights and biases
                new_conv_layer.weight.data = pruned_weights
                if module.bias is not None:
                    new_conv_layer.bias.data = module.bias.data[keep_indices]

                # Replace the original layer with the pruned layer
                model._modules[list(model._modules.keys())[i]] = new_conv_layer

    print(f"Pruning aplikovan. Iterace {iteration}, váhy odstraněny.\n")

    # Update pruning ratio
    current_pruning *= (1 - pruning_rate)
    print(f"Podil vah: {current_pruning * 100:.2f} %\n")

# Save pruned model
pruned_model_path = 'runs/train/exp2/weights/pruned_model.pt'
torch.save({'model': model}, pruned_model_path)

# Print the model structure after pruning
print(f"Model po pruning: {model}")
