import torch
from torch import nn, optim
import sys
import torch
import os
import time
import argparse
from utils.progress_bar import progress_bar
from utils.average_meter import AverageMeter


class GradientAscent:

    def __init__(self, model, LOADER_DICT, args):
        self.model = model
        self.train_loader = LOADER_DICT['forget']
        self.remained_loader = LOADER_DICT['remain']
        self.test_loader = LOADER_DICT['test']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= args.unlearn_epochs)
        self.device = args.device
        self.num_epochs = args.unlearn_epochs
        self.checkpoint_dir = args.checkpoint_dir
        self.best_acc = 0
        self.args = args

    def gradient_ascent(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss_meter = AverageMeter()
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = -self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss_meter.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                              % (train_loss_meter.avg, 100. * correct / total, correct, total))

    def unlearn(self):
        total_time = 0
        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.gradient_ascent(epoch)
            self.scheduler.step()
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time
        print(f"Total time for {self.num_epochs} epochs: {total_time:.2f} seconds.")
        return self.model


class GradientAscentPlus:  # NegGrad+

    def __init__(self, model, LOADER_DICT, args):
        self.model = model
        self.forget_loader = LOADER_DICT['forget']  # Dataset to unlearn
        self.remained_loader = LOADER_DICT['remain']  # Dataset to retain accuracy
        self.test_loader = LOADER_DICT['test']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.unlearn_epochs)
        self.device = args.device
        self.num_epochs = args.unlearn_epochs
        self.refine_epochs = args.refine_epochs  # New step for refining with 'remain'
        self.checkpoint_dir = args.checkpoint_dir
        self.best_acc = 0
        self.args = args

    def gradient_ascent(self, epoch):
        """
        Perform Negative Gradient Ascent for the forget set.
        """
        print(f'\nEpoch (Gradient Ascent): {epoch + 1}')
        self.model.train()
        train_loss_meter = AverageMeter()
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.forget_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = -self.criterion(outputs, targets)  # Negative loss to unlearn
            loss.backward()
            self.optimizer.step()

            train_loss_meter.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.forget_loader),
                         f'Unlearn Loss: {train_loss_meter.avg:.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total})')

    def refine_remained(self, epoch):
        """
        Fine-tune the model on the 'remain' set to retain performance.
        """
        print(f'\nEpoch (Refinement): {epoch + 1}')
        self.model.train()
        refine_loss_meter = AverageMeter()
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.remained_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)  # Standard loss for refinement
            loss.backward()
            self.optimizer.step()
            refine_loss_meter.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.remained_loader),
                         f'Refine Loss: {refine_loss_meter.avg:.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total})')

    def test(self):
        """
        Evaluate the model on the test dataset.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        print(f'\nTest Loss: {test_loss / len(self.test_loader):.3f} | Test Accuracy: {acc:.3f}%')
        return acc

    def unlearn(self):
        """
        Perform NegGrad+ unlearning with refinement.
        """
        total_time = 0

        # Step 1: Negative Gradient Ascent for Forgetting
        print("\n--- Step 1: Performing Gradient Ascent (Unlearning) ---")
        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.gradient_ascent(epoch)
            self.scheduler.step()
            end_time = time.time()
            total_time += (end_time - start_time)

        # Step 2: Refinement on Remained Set
        print("\n--- Step 2: Refining Model on Remained Dataset ---")
        for epoch in range(self.refine_epochs):
            start_time = time.time()
            self.refine_remained(epoch)
            self.scheduler.step()
            end_time = time.time()
            total_time += (end_time - start_time)

        # Final Testing
        print("\n--- Step 3: Evaluating Model on Test Dataset ---")
        acc = self.test()
        print(f"\nTotal Time for Unlearning and Refinement: {total_time:.2f} seconds.")
        print(f"Final Test Accuracy after Unlearning and Refinement: {acc:.3f}%")
        return self.model

