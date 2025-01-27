import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


sys.path.append(".")


def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()

def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        optimizer = torch.optim.SGD(
             model.parameters(),
             args.unlearn_lr,
             momentum=args.momentum,
             weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=0.1
        )  # 0.1 is fixed

        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()
            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )
            train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args
            )
            scheduler.step()
            print("one epoch duration:{}".format(time.time() - start_time))

    return _wrapped


def iterative_unlearn(func):

    return _iterative_unlearn_impl(func)