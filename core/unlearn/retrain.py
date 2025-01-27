import torch
import torch.nn as nn
import os
import time
import argparse
from utils.progress_bar import progress_bar
from utils.average_meter import AverageMeter


class Retrain:

    def __init__(self, model, LOADER_DICT, args):
        self.model = model
        self.train_loader = LOADER_DICT['remain']
        self.test_loader = LOADER_DICT['test']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=280)
        self.device = args.device
        self.num_epochs = args.unlearn_epochs
        self.checkpoint_dir = args.checkpoint_dir
        self.best_acc = 0
        self.args = args
        self.save = False
        # if self.args.certified exists, then set self.certified to True
        if hasattr(self.args, 'certified'):
            self.certified = self.args.certified

    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss_meter = AverageMeter()
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss_meter.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #                   % (train_loss_meter.avg, 100. * correct / total, correct, total))

            if self.certified == True:
                max_norm = 20
                param_norm = nn.utils.parameters_to_vector(self.model.parameters()).norm()
                if param_norm > max_norm:
                    scale_factor = max_norm / param_norm
                    for param in self.model.parameters():
                        param.data *= scale_factor



    def test(self, epoch):
        self.model.eval()
        test_loss_meter = AverageMeter()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss_meter.update(loss.item(), inputs.size(0))
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(self.test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #                   % (test_loss_meter.avg, 100. * correct / total, correct, total))

        acc = 100. * correct / total

        if acc > self.best_acc:
            self.best_acc = acc
            if self.save:
                self.save_checkpoint(epoch, acc)

    def save_checkpoint(self, epoch, acc):
        print('Saving..')
        state = {
            'model_state_dict': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        torch.save(state, os.path.join(self.checkpoint_dir, f'retrain_{self.args.dataset}_{self.args.seed}_{self.args.forget_size}_{self.args.task}.pth'))

    # @staticmethod
    # def progress_bar(current, total, msg):
    #     print(f'{current}/{total}: {msg}')

    def unlearn(self):
        total_time = 0
        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time
        print(f"Total time for {self.num_epochs} epochs: {total_time:.2f} seconds.")


