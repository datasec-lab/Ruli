import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import time
import argparse
import copy
import numpy as np
import sys
import torch
from torch import nn
import wandb
import torch.optim as optim
import argparse as a


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network using KL Divergence."""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


class Scrub:

    def __init__(self, model, LOADER_DICT, args):
        self.model = model
        self.alpha = args.scrub_alpha
        self.beta = args.scrub_beta
        self.gamma = args.scrub_gamma
        self.m_steps = args.m_steps
        self.data_loaders = LOADER_DICT
        self.device = args.device
        self.T = args.T
        self.optim = args.optimizer
        self.smoothing = args.smoothing
        self.lr_decay_epochs = args.lr_decay_epochs
        self.lr_decay_rate = args.lr_decay_rate
        self.args = args

    def unlearn(self):


        forget_loader = self.data_loaders['forget']
        retain_loader = self.data_loaders['remain']
        valid_loader_full = self.data_loaders['test']

        teacher = copy.deepcopy(self.model)
        student = copy.deepcopy(self.model)
        model_t = copy.deepcopy(teacher)
        model_s = copy.deepcopy(student)

        module_list = nn.ModuleList([])
        module_list.append(model_s)
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_s)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(self.T)
        criterion_kd = DistillKL(self.T)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)  # other knowledge distillation loss

        # optimizer
        if self.optim == "sgd":
            optimizer = optim.SGD(trainable_list.parameters(),
                                  lr=self.args.lr,
                                  momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay)
        elif self.optim == "adam":
            optimizer = optim.Adam(trainable_list.parameters(),
                                   lr=self.args.lr,
                                   weight_decay=self.args.weight_decay)
        elif self.optim == "rmsp":
            optimizer = optim.RMSprop(trainable_list.parameters(),
                                      lr=self.args.lr,
                                      momentum=self.args.momentum,
                                      weight_decay=self.args.weight_decay)

        module_list.append(model_t)

        if torch.cuda.is_available():
            module_list.cuda()
            criterion_list.cuda()
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True

        # send to args.device (cuda or cpu)
        # if torch.cuda.is_available():
        #     module_list.to(self.device)
        #     criterion_list.to(self.device)
        #     import torch.backends.cudnn as cudnn
        #     cudnn.benchmark = True

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            return (1 - self.beta) * averaged_model_parameter + self.beta * model_parameter

        swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)
        swa_model.to(self.device)

        for epoch in range(1, self.args.unlearn_epochs + 1):
            lr_decay_epochs = [3, 5, 9]
            lr, optimizer = adjust_learning_rate(epoch, self.args.lr, lr_decay_epochs, self.lr_decay_rate, optimizer)
            maximize_loss = 0
            if epoch <= self.m_steps:
                maximize_loss = train_distill(forget_loader, module_list, swa_model,
                                              criterion_list, optimizer, self.args, "maximize",
                                              quiet=False)

            train_acc, train_loss = train_distill(retain_loader, module_list, swa_model, criterion_list,
                                                  optimizer, self.args,
                                                  "minimize", quiet=False)

            print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss,
                                                                                         train_loss, train_acc))

        return model_s


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist


def train_distill(train_loader, module_list, swa_model, criterion_list, optimizer, args, split, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        input, target = data
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        #feat_s, logit_s = model_s(input, is_feat=True, preact=False)
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0

        if split == "minimize":
            loss = args.scrub_gamma * loss_cls + args.scrub_alpha * loss_div + args.scrub_beta * loss_kd
        elif split == "maximize":
            loss = -loss_div

        loss = loss + param_dist(model_s, swa_model, args.smoothing)

        if split == "minimize" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(model_s.parameters(), clip)
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    if split == "minimize":
        if not quiet:
            print(' * Acc@1 {top1.avg:.3f} '
                  .format(top1=top1))

        return top1.avg, losses.avg
    else:
        return kd_losses.avg


def adjust_learning_rate(epoch, learning_rate, lr_decay_epochs, lr_decay_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = learning_rate
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr, optimizer


def validate(val_loader, model, criterion, opt, quiet=False):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not quiet:
                if idx % opt.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        idx, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))
        if not quiet:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg