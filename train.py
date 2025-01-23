import torch
import torch.nn as nn
import os
import time
import argparse
from utils.progress_bar import progress_bar
from utils.average_meter import AverageMeter
from tqdm import tqdm


class Retrain:

    def __init__(self, model, LOADER_DICT, args):
        self.model = model
        self.train_loader = LOADER_DICT['train']
        self.test_loader = LOADER_DICT['test']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.train_epochs)
        self.device = args.device
        self.num_epochs = args.train_epochs
        self.checkpoint_dir = args.checkpoint_dir
        self.best_acc = 0
        self.args = args

    from tqdm import tqdm

    def train(self, epoch, progress):
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

            # Update the progress bar after each batch
        # progress.set_postfix({
        #     'Epoch': epoch + 1,
        #     'Train Loss': f'{train_loss_meter.avg:.3f}',
        #     'Train Acc': f'{100. * correct / total:.3f}%'})

        return train_loss_meter.avg, correct, total

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
        torch.save(state, os.path.join(self.checkpoint_dir,
                                       f'retrain_{self.args.dataset}_{self.args.seed}_{self.args.forget_size}.pth'))

    def test(self, epoch, progress):
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

                #Update the progress bar after each batch
                # progress.set_postfix({
                #     'Epoch': epoch + 1,
                #     'Test Loss': f'{test_loss_meter.avg:.3f}',
                #     'Test Acc': f'{100. * correct / total:.3f}%'})
                # progress.update(1)

        acc = 100. * correct / total
        if acc > self.best_acc:
            self.best_acc = acc
            #self.save_checkpoint(epoch, acc)

        return test_loss_meter.avg, correct, total

    def unlearn(self):
        total_time = 0
        total_batches = self.num_epochs * (len(self.train_loader) + len(self.test_loader))

        # Initialize the progress bar
        with tqdm(total=total_batches, desc="Training Progress", unit="batch") as progress:
            for epoch in range(self.num_epochs):
                start_time = time.time()

                train_loss_avg, train_correct, train_total = self.train(epoch, progress)
                test_loss_avg, test_correct, test_total = self.test(epoch, progress)

                self.scheduler.step()

                end_time = time.time()
                epoch_time = end_time - start_time
                total_time += epoch_time

                # Update final progress bar details at the end of each epoch
        # progress.set_postfix({
        #     'Train Loss': f'{train_loss_avg:.3f}',
        #     'Train Acc': f'{100. * train_correct / train_total:.3f}%',
        #     'Test Loss': f'{test_loss_avg:.3f}',
        #     'Test Acc': f'{100. * test_correct / test_total:.3f}%'
        # })

        print(f"Total time for {self.num_epochs} epochs: {total_time:.2f} seconds.")