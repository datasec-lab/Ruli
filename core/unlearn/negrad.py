import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from torch.autograd import grad
import time
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from torch.autograd import grad
import time
# add progress bar
import torch
from torch import nn, optim
from itertools import cycle
from tqdm import tqdm
import time
from utils.progress_bar import progress_bar
from utils.average_meter import AverageMeter


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class NegGrad:


    def __init__(self, model, LOADER_DICT, args):
        self.model = model
        self.model_init = model
        self.data_loader = LOADER_DICT['remain']
        self.forget_loader = LOADER_DICT['forget']
        self.test_loader = LOADER_DICT['test']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.unlearn_epochs)
        self.device = args.device
        self.num_epochs = args.unlearn_epochs
        self.alpha = args.alpha
        self.weight_decay = args.weight_decay
        self.args = args

    @staticmethod
    def l2_penalty(model, model_init, weight_decay):
        l2_loss = 0
        for (k, p), (k_init, p_init) in zip(model.named_parameters(), model_init.named_parameters()):
            if p.requires_grad:
                l2_loss += (p - p_init).pow(2).sum()
        l2_loss *= (weight_decay / 2.)
        return l2_loss

    def run_train_epoch(self, split: str, epoch: int, quiet: bool = False):
        print(f'\nEpoch ({split}): {epoch + 1}')
        self.model.eval() if split == 'test' else self.model.train()
        metrics = AverageMeter()

        with torch.set_grad_enabled(split != 'test'):
            for idx, batch in enumerate(tqdm(self.data_loader, leave=False)):
                batch = [tensor.to(self.device) for tensor in batch]
                inputs, targets = batch
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets) + NegGrad.l2_penalty(self.model, self.model_init, self.weight_decay)
                #metrics.update(n=inputs.size(0), loss=self.criterion(outputs, targets).item())

                if split != 'test':
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                progress_bar(idx, len(self.data_loader), f'Loss: {metrics.avg:.3f}')

        if not quiet:
            print(f'{split.capitalize()} Metrics - Loss: {metrics.avg:.3f}')
        return metrics.avg

    def run_neggrad_epoch(self, epoch: int, quiet: bool = False):
        print(f'\nEpoch (NegGrad): {epoch + 1}')
        self.model.train()
        metrics = AverageMeter()

        with torch.set_grad_enabled(True):
            for idx, (retain_batch, forget_batch) in enumerate(tqdm(zip(cycle(self.data_loader), self.forget_loader), leave=False)):
                retain_batch = [tensor.to(self.device) for tensor in retain_batch]
                forget_batch = [tensor.to(self.device) for tensor in forget_batch]

                inputs_r, targets_r = retain_batch
                inputs_f, targets_f = forget_batch

                outputs_r = self.model(inputs_r)
                outputs_f = self.model(inputs_f)

                loss = self.alpha * (self.criterion(outputs_r, targets_r) + NegGrad.l2_penalty(self.model, self.model_init, self.weight_decay)) - (1 - self.alpha) * self.criterion(outputs_f, targets_f)
                #metrics.update(n=inputs_r.size(0), loss=self.criterion(outputs_r, targets_r).item())

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress_bar(idx, len(self.data_loader), f'NegGrad Loss: {metrics.avg:.3f}')

        if not quiet:
            print(f'NegGrad Metrics - Loss: {metrics.avg:.3f}')
        return metrics.avg

    def test(self):
        print("\n--- Testing Model ---")
        self.model.eval()
        metrics = AverageMeter()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                #metrics.update(n=inputs.size(0), loss=loss.item())
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        print(f'Test Loss: {metrics.avg:.3f} | Test Accuracy: {acc:.3f}%')
        return acc

    def unlearn(self):
        total_time = 0

        # Step 1: NegGrad Epochs
        print("\n--- Step 1: Performing NegGrad Epochs ---")
        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.run_neggrad_epoch(epoch)
            self.scheduler.step()
            end_time = time.time()
            total_time += (end_time - start_time)

        # Step 2: Testing
        print("\n--- Step 2: Testing ---")
        final_acc = self.test()

        print(f"\nTotal Time for Unlearning: {total_time:.2f} seconds.")
        print(f"Final Test Accuracy after Unlearning: {final_acc:.3f}%")
        return self.model
