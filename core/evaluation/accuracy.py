import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def eval_accuracy(model, data_loader, device='cuda'):
    model.eval()  # Switch the model to evaluation mode.
    device = torch.device(device)
    total_hits = 0
    total_samples = 0
    mode = 'backdoor'
    print_perform = False
    name = ''

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            if mode == 'pruned':
                output = output[:, 0:10]  # Assuming the 'pruned' mode requires adjusting the output.
            predictions = torch.argmax(output, dim=1)
            total_hits += (predictions == batch_y).sum().item()
            total_samples += batch_y.size(0)
    accuracy = total_hits / total_samples
    if print_perform:
        print(f"Model '{name}' Accuracy: {accuracy*100:.2f}%")
    return accuracy


def distance(model,model0):
    distance=0
    normalization=0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        space='  ' if 'bias' in k else ''
        current_dist=(p.data0-p0.data0).pow(2).sum().item()
        current_norm=p.data0.pow(2).sum().item()
        distance+=current_dist
        normalization+=current_norm
    print(f'Distance: {np.sqrt(distance)}')
    print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    return 1.0*np.sqrt(distance/normalization)