import torch
from copy import deepcopy

# load
model_path = 'runs/train/exp2/weights/best.pt'
model = torch.load(model_path, map_location='cuda')['model'].float()

# original layers for restart
conv_layers = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
original_weights = {id(module): deepcopy(module.weight.data) for module in conv_layers}

# parameters for pruning
pruning_rate = 0.2  # 20% treshold
target_pruning = 0.1  # 10% to keep
current_pruning = 1.0  

iteration = 0

# itterative prunninigs
while current_pruning > target_pruning:
    iteration += 1
    print(f"== Iterace {iteration} ==")

    
    all_weights = torch.cat([module.weight.data.view(-1) for module in conv_layers])
    
    
    threshold = torch.quantile(all_weights.abs(), pruning_rate)
    print(f"Threshold pro iteraci {iteration}: {threshold.item()}")

    # apply mask
    with torch.no_grad():
        for module in conv_layers:
            weights = module.weight.data
            mask = (weights.abs() >= threshold).float()  # 1 pro váhy >= threshold, 0 jinak
            module.weight.data = weights * mask  # Nastav malé váhy na nulu

    print("Pruning aplikovan.")

    # RESET
    print("Reset")
    with torch.no_grad():
        for module in conv_layers:
            original_weight = original_weights[id(module)]
            mask = (module.weight.data != 0).float()  # Zachované váhy
            module.weight.data = mask * original_weight  # Reset váhy na původní hodnoty

    print("Reseted.")

    # Update
    current_pruning *= (1 - pruning_rate)
    print(f"Podil vah: {current_pruning * 100:.2f} %\n")


pruned_model_path = 'runs/train/exp2/weights/pruned_model.pt'
torch.save({'model': model}, pruned_model_path)

# ... to be trained
