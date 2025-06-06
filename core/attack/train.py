import torch
import torch.nn as nn
import os
import time
import argparse
from utils.progress_bar import progress_bar
from utils.average_meter import AverageMeter
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy



# class Retrain:
#
#     def __init__(self, model, LOADER_DICT, args):
#         self.model = model
#         self.train_loader = LOADER_DICT['train']
#         self.test_loader = LOADER_DICT['test']
#         self.criterion = nn.CrossEntropyLoss()
#
#         if args.arch == 'vit':
#             self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#         else:
#             self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
#                                              weight_decay=args.weight_decay)
#
#         # self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
#         #                                  weight_decay=args.weight_decay)
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.train_epochs)
#         self.device = args.device
#         self.num_epochs = args.train_epochs
#         self.checkpoint_dir = args.checkpoint_dir
#         self.best_acc = 0
#         self.args = args
#
#     from tqdm import tqdm
#
#     def train(self, epoch, progress):
#         self.model.train()
#         train_loss_meter = AverageMeter()
#         correct = 0
#         total = 0
#
#         for batch_idx, (inputs, targets) in enumerate(self.train_loader):
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
#             self.optimizer.zero_grad()
#             outputs = self.model(inputs)
#             loss = self.criterion(outputs, targets)
#             loss.backward()
#             self.optimizer.step()
#             train_loss_meter.update(loss.item(), inputs.size(0))
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#             # Update the progress bar after each batch
#         progress.set_postfix({
#             'Epoch': epoch + 1,
#             'Train Loss': f'{train_loss_meter.avg:.3f}',
#             'Train Acc': f'{100. * correct / total:.3f}%'})
#
#         progress.update(1)
#
#         return train_loss_meter.avg, correct, total
#
#     def save_checkpoint(self, epoch, acc):
#         print('Saving..')
#         state = {
#             'model_state_dict': self.model.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#             'optimizer_state_dict': self.optimizer.state_dict(),
#         }
#         if not os.path.isdir(self.checkpoint_dir):
#             os.mkdir(self.checkpoint_dir)
#         torch.save(state, os.path.join(self.checkpoint_dir,
#                                        f'retrain_{self.args.dataset}_{self.args.seed}_{self.args.forget_size}.pth'))
#
#     def test(self, epoch, progress):
#         self.model.eval()
#         test_loss_meter = AverageMeter()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch_idx, (inputs, targets) in enumerate(self.test_loader):
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, targets)
#                 test_loss_meter.update(loss.item(), inputs.size(0))
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
#
#                 #Update the progress bar after each batch
#                 # progress.set_postfix({
#                 #     'Epoch': epoch + 1,
#                 #     'Test Loss': f'{test_loss_meter.avg:.3f}',
#                 #     'Test Acc': f'{100. * correct / total:.3f}%'})
#                 # progress.update(1)
#
#         acc = 100. * correct / total
#         if acc > self.best_acc:
#             self.best_acc = acc
#             #self.save_checkpoint(epoch, acc)
#
#         return test_loss_meter.avg, correct, total
#
#     def unlearn(self):
#         total_time = 0
#         total_batches = self.num_epochs * (len(self.train_loader) + len(self.test_loader))
#
#         # Initialize the progress bar
#         with tqdm(total=total_batches, desc="Training Progress", unit="batch") as progress:
#             for epoch in range(self.num_epochs):
#                 start_time = time.time()
#
#                 train_loss_avg, train_correct, train_total = self.train(epoch, progress)
#                 test_loss_avg, test_correct, test_total = self.test(epoch, progress)
#                 #print("Test accuracy: ", test_correct / test_total)
#
#                 self.scheduler.step()
#                 end_time = time.time()
#                 epoch_time = end_time - start_time
#                 total_time += epoch_time
#
#                 # Update final progress bar details at the end of each epoch
#         progress.set_postfix({
#             'Train Loss': f'{train_loss_avg:.3f}',
#             'Train Acc': f'{100. * train_correct / train_total:.3f}%',
#             'Test Loss': f'{test_loss_avg:.3f}',
#             'Test Acc': f'{100. * test_correct / test_total:.3f}%'
#         })
#         progress.update(1)
#
#         print(f"Total time for {self.num_epochs} epochs: {total_time:.2f} seconds.")

import torch
import torch.nn as nn
import time
import os
from tqdm import tqdm

class Retrain:

    def __init__(self, model, LOADER_DICT, args):
        self.model = model
        self.train_loader = LOADER_DICT['train']
        self.test_loader = LOADER_DICT['test']
        args.is_multilabel = False

        if args.is_multilabel:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        if args.arch in ['vit', 'bert-classifier', 'roberta-classifier']:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                             weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.train_epochs)
        self.device = args.device
        self.num_epochs = args.train_epochs
        self.checkpoint_dir = args.checkpoint_dir
        self.best_acc = 0
        self.args = args

    def train(self, epoch, progress):
        self.model.train()
        train_loss_meter = AverageMeter()
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            if isinstance(batch, dict):  # For text models
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:  # For image models
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            train_loss_meter.update(loss.item(), labels.size(0))
            _, predicted = logits.max(1)
            if self.args.is_multilabel:
                predicted = (torch.sigmoid(logits) > 0.5).long()

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item() if not self.args.is_multilabel else (predicted == labels).sum().item()

        progress.set_postfix({
            'Epoch': epoch + 1,
            'Train Loss': f'{train_loss_meter.avg:.3f}',
            'Train Acc': f'{100. * correct / total:.3f}%'
        })
        progress.update(1)

        return train_loss_meter.avg, correct, total

    def save_checkpoint(self, epoch, acc):
        print('Saving checkpoint...')
        state = {
            'model_state_dict': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(state, os.path.join(self.checkpoint_dir,
                                       f'retrain_{self.args.dataset}_{self.args.seed}_{self.args.forget_size}.pth'))

    def test(self, epoch, progress):
        self.model.eval()
        test_loss_meter = AverageMeter()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if isinstance(batch, dict):  # Text models
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                else:  # Image models
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    logits = self.model(inputs)

                loss = self.criterion(logits, labels)
                test_loss_meter.update(loss.item(), labels.size(0))
                _, predicted = logits.max(1)
                if self.args.is_multilabel:
                    predicted = (torch.sigmoid(logits) > 0.5).long()

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item() if not self.args.is_multilabel else (predicted == labels).sum().item()

        acc = 100. * correct / total
        if acc > self.best_acc:
            self.best_acc = acc
            # Uncomment if you want automatic saving:
            # self.save_checkpoint(epoch, acc)

        return test_loss_meter.avg, correct, total

    def unlearn(self):
        total_time = 0
        total_batches = self.num_epochs * (len(self.train_loader) + len(self.test_loader))

        with tqdm(total=total_batches, desc="Training Progress", unit="batch") as progress:
            for epoch in range(self.num_epochs):
                start_time = time.time()

                train_loss_avg, train_correct, train_total = self.train(epoch, progress)
                test_loss_avg, test_correct, test_total = self.test(epoch, progress)

                self.scheduler.step()
                end_time = time.time()
                epoch_time = end_time - start_time
                total_time += epoch_time

        progress.set_postfix({
            'Train Loss': f'{train_loss_avg:.3f}',
            'Train Acc': f'{100. * train_correct / train_total:.3f}%',
            'Test Loss': f'{test_loss_avg:.3f}',
            'Test Acc': f'{100. * test_correct / test_total:.3f}%'
        })
        progress.update(1)

        print(f"Total time for {self.num_epochs} epochs: {total_time:.2f} seconds.")


import math
#
# class RetrainText:
#     def __init__(self, model, LOADER_DICT, args):
#         self.model = model
#         self.train_loader = LOADER_DICT['train']
#         self.test_loader = LOADER_DICT['test']
#         self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')  # token-level loss
#
#         self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.train_epochs)
#         self.device = args.device
#         self.num_epochs = args.train_epochs
#         self.checkpoint_dir = args.checkpoint_dir
#         self.best_loss = float('inf')
#         self.args = args
#
#     def train(self, epoch, progress):
#         self.model.train()
#         train_loss_meter = AverageMeter()
#         total_tokens = 0
#
#         for batch in self.train_loader:
#             input_ids = batch['input_ids'].to(self.device)
#             attention_mask = batch['attention_mask'].to(self.device)
#             labels = input_ids.clone()
#
#             outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#             train_loss_meter.update(loss.item(), input_ids.size(0))
#             total_tokens += attention_mask.sum().item()
#
#         avg_loss = train_loss_meter.avg
#         avg_ppl = math.exp(avg_loss)
#
#         progress.set_postfix({
#             'Epoch': epoch + 1,
#             'Train Loss': f'{avg_loss:.3f}',
#             'Train PPL': f'{avg_ppl:.2f}'
#         })
#         progress.update(1)
#
#         return avg_loss, avg_ppl, total_tokens
#
#     def test(self, epoch, progress):
#         self.model.eval()
#         val_loss_meter = AverageMeter()
#         total_tokens = 0
#
#         with torch.no_grad():
#             for batch in self.test_loader:
#                 input_ids = batch['input_ids'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = input_ids.clone()
#                 labels = input_ids.clone()
#                 labels[attention_mask == 0] = -100
#
#                 outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
#                 loss = outputs.loss
#
#                 val_loss_meter.update(loss.item(), input_ids.size(0))
#                 total_tokens += attention_mask.sum().item()
#
#         avg_loss = val_loss_meter.avg
#         avg_ppl = math.exp(avg_loss)
#
#         if avg_loss < self.best_loss:
#             self.best_loss = avg_loss
#             self.save_checkpoint(epoch, avg_loss)
#
#         return avg_loss, avg_ppl, total_tokens
#
#     def save_checkpoint(self, epoch, val_loss):
#         # Save checkpoint if needed (optional, not activated here)
#         pass
#
#     def unlearn(self):
#         total_time = 0
#         total_batches = self.num_epochs * (len(self.train_loader) + len(self.test_loader))
#
#         with tqdm(total=total_batches, desc="Training Progress", unit="batch") as progress:
#             for epoch in range(self.num_epochs):
#                 start_time = time.time()
#
#                 train_loss_avg, train_ppl, train_tokens = self.train(epoch, progress)
#                 val_loss_avg, val_ppl, val_tokens = self.test(epoch, progress)
#
#                 self.scheduler.step()
#                 end_time = time.time()
#                 epoch_time = end_time - start_time
#                 total_time += epoch_time
#
#                 print(f"Epoch {epoch + 1}/{self.num_epochs} Summary:")
#                 print(f"  Train Loss: {train_loss_avg:.4f}, Train Perplexity: {train_ppl:.2f}")
#                 print(f"  Val Loss:   {val_loss_avg:.4f}, Val Perplexity:   {val_ppl:.2f}")
#                 print(f"  Epoch Time: {epoch_time:.2f}s\n")
#
#         print(f"Total time for {self.num_epochs} epochs: {total_time:.2f} seconds.")
#         print(f"Best validation loss: {self.best_loss:.4f}")

#TODO: code to accelerate


import torch
import torch.nn as nn
import math
import time
from tqdm import tqdm
import torch.backends.cudnn as cudnn

class AverageMeter:
    """Utility class to track average values."""
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

class RetrainText:
    def __init__(self, model, LOADER_DICT, args):
        self.model = model.to(args.device)
        self.train_loader = LOADER_DICT['train']
        self.test_loader = LOADER_DICT['test']
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.train_epochs)
        self.device = args.device
        self.num_epochs = args.train_epochs
        self.checkpoint_dir = args.checkpoint_dir
        self.best_loss = float('inf')
        self.args = args
        self.scaler = torch.cuda.amp.GradScaler()

        # Enable cuDNN auto-tuning
        cudnn.benchmark = True

    def train(self, epoch, progress):
        self.model.train()
        train_loss_meter = AverageMeter()
        total_tokens = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = input_ids.clone()

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss_meter.update(loss.item(), input_ids.size(0))
            total_tokens += attention_mask.sum().item()

        avg_loss = train_loss_meter.avg
        avg_ppl = math.exp(avg_loss)

        progress.set_postfix({
            'Epoch': epoch + 1,
            'Train Loss': f'{avg_loss:.3f}',
            'Train PPL': f'{avg_ppl:.2f}'
        })
        progress.update(1)

        return avg_loss, avg_ppl, total_tokens

    def test(self, epoch, progress):
        self.model.eval()
        val_loss_meter = AverageMeter()
        total_tokens = 0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                val_loss_meter.update(loss.item(), input_ids.size(0))
                total_tokens += attention_mask.sum().item()

        avg_loss = val_loss_meter.avg
        avg_ppl = math.exp(avg_loss)

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_checkpoint(epoch, avg_loss)

        return avg_loss, avg_ppl, total_tokens

    def save_checkpoint(self, epoch, val_loss):
        # You can implement checkpoint saving here if needed
        pass

    def unlearn(self):
        total_time = 0
        total_batches = self.num_epochs * (len(self.train_loader) + len(self.test_loader))

        with tqdm(total=total_batches, desc="Training Progress", unit="batch") as progress:
            for epoch in range(self.num_epochs):
                start_time = time.time()

                train_loss_avg, train_ppl, _ = self.train(epoch, progress)
                val_loss_avg, val_ppl, _ = self.test(epoch, progress)

                self.scheduler.step()
                epoch_time = time.time() - start_time
                total_time += epoch_time

                print(f"Epoch {epoch + 1}/{self.num_epochs} Summary:")
                print(f"  Train Loss: {train_loss_avg:.4f}, Train Perplexity: {train_ppl:.2f}")
                print(f"  Val Loss:   {val_loss_avg:.4f}, Val Perplexity:   {val_ppl:.2f}")
                print(f"  Epoch Time: {epoch_time:.2f}s\n")

        print(f"Total time for {self.num_epochs} epochs: {total_time:.2f} seconds.")
        print(f"Best validation loss: {self.best_loss:.4f}")
