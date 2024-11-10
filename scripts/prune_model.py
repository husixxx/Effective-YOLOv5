import torch
import torch.nn.utils.prune as prune

# Načítaj model
model = torch.load('runs/train/exp2/weights/best.pt', map_location='cuda')['model'].float()  # Načítanie modelu

# Práh pruningu
pruning_rate = 0.15  # Znížený prah pruningu na 15%

# Zoznam konvolučných vrstiev (uložíme všetky názvy konvolučných vrstiev, aby sme vedeli, ktoré sú posledné)
conv_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]

# Aplikuj pruning len na posledné konvolučné vrstvy
num_layers_to_prune = 5  # Počet posledných konvolučných vrstiev, ktoré chceme prunovať
for name in conv_layers[-num_layers_to_prune:]:  # Vyberie len posledných 5 vrstiev
    module = dict(model.named_modules())[name]
    prune.l1_unstructured(module, name='weight', amount=pruning_rate)

# Odstráň masky pre lepšiu inferenciu
for name in conv_layers[-num_layers_to_prune:]:  # Znova len na posledných 5 vrstvách
    module = dict(model.named_modules())[name]
    prune.remove(module, 'weight')

# Ulož pruned model
torch.save({'model': model}, 'runs/train/exp2/pruned_best.pt')

