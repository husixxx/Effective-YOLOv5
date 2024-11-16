import torch
import torch.nn.utils.prune as prune

model = torch.load('runs/train/exp2/weights/best.pt', map_location='cuda')['model'].float()


pruning_rate = 0.15


conv_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]


num_layers_to_prune = 5
for name in conv_layers[-num_layers_to_prune:]:
    module = dict(model.named_modules())[name]
    prune.l1_unstructured(module, name='weight', amount=pruning_rate)

for name in conv_layers[-num_layers_to_prune:]:  
    module = dict(model.named_modules())[name]
    prune.remove(module, 'weight')

torch.save({'model': model}, 'runs/train/exp2/pruned_best.pt')

