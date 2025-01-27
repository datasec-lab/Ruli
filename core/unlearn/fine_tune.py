import sys
import time
import torch
from utils.average_meter import AverageMeter
sys.path.append(".")


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class FineTune:

    def __init__(self, model, LOADER_DICT, args):

        self.retain_loader = LOADER_DICT['remain']
        self.test_loader = LOADER_DICT['test']
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                         momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.unlearn_epochs)
        self.device = args.device
        self.num_epochs = args.unlearn_epochs
        self.with_l1 = args.with_l1
        self.alpha = args.sparse_alpha
        self.no_l1_epochs = args.no_l1_epochs
        self.args = args
        self.sparse_scheduler = args.sparse_scheduler

    @staticmethod
    def calculate_sparsity(model):
        total_params = 0
        zero_params = 0
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param.abs() < 1e-6).sum().item()  # Count parameters close to zero
        return zero_params / total_params

    def unlearn(self):
        for epoch in range(self.num_epochs):
            if epoch <= self.num_epochs - self.no_l1_epochs:
                if self.sparse_scheduler == 'decay':
                    current_alpha = (2 - (2 * epoch / self.num_epochs)) * self.alpha
                elif self.sparse_scheduler == 'constant':
                    current_alpha = self.alpha
                elif self.sparse_scheduler == 'increase':
                    current_alpha = (2 * epoch / self.num_epochs) * self.alpha
            elif epoch > self.num_epochs - self.no_l1_epochs:
                current_alpha = 0
            self.fine_tune(current_alpha)

        sparsity = FineTune.calculate_sparsity(self.model)
        print(f"Sparsity achieved: {sparsity * 100:.2f}%" , "sparsity scheduler", self.sparse_scheduler)
        return self.model

    def fine_tune(self, current_alpha):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        start = time.time()

        for i, (image, target) in enumerate(self.retain_loader):
            image, target = image.to(self.device), target.to(self.device)

            output = self.model(image)
            loss = self.criterion(output, target)
            if self.with_l1:
                loss += current_alpha * l1_regularization(self.model)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

        print(f"train_accuracy {top1.avg:.3f}")
        self.scheduler.step()




