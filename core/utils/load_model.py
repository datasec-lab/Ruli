
from torch import nn
import torch
from models import ResNet18, ResNet18_cifar100, vgg16_bn, wrn28_10, Resnext50
from torchvision.models import resnet18, resnet50, vgg16_bn
import timm
import random
import numpy as np
import torchvision.models as torch_models
import torch
import torch.nn as nn
import random
import numpy as np
import time


def seed_everything(seed=None):
    """
    Function to set the random seed for reproducibility across all necessary libraries.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # No seed provided, so randomness will be system-dependent
        torch.backends.cudnn.benchmark = True  # Allows non-deterministic algorithms for better performance


def prepare_torchvision_model(args):
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'TinyImageNet':
        num_classes = 200

    if args.arch == 'resnet18':
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=False)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=False)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model.to(args.device)
    return model


# def prepare_compression_model(args):
#
#     if args.arch == 'resnet18_filtered':
#         model = ResNet18Filtered(num_classes=10)
#     else:
#         raise ValueError(f"Unknown architecture: {args.arch}")
#     model.to(args.device)
#     return model


def prepare_model(args, fresh_seed=True):
    """
    Prepares the model by initializing it and optionally loading from a checkpoint,
    with a fresh seed for each model preparation.
    """
    if not fresh_seed:
        fresh_seed = args.seed

    else:
        fresh_seed = int(time.time() * 1000) % (2 ** 32 - 1)

    seed_everything(fresh_seed)
    model = load_model(args)
    if args.trained_model_path is not None:
        model = load_from_checkpoint(model, args.trained_model_path, args.device)
    else:
        model.to(args.device)
    return model

def load_model(args):
    """
    Function to load a model architecture based on the arguments provided.
    """
    num_classes = 10
    model = None

    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'TinyImageNet':
        num_classes = 200


    # Vision models
    if args.arch == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(num_classes=num_classes)
    elif args.arch == 'wrn28_10':
        model = wrn28_10(num_classes=num_classes)
    elif args.arch == 'vit':
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, drop_path_rate=0.1)
        model.reset_classifier(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    return model




@staticmethod
def load_from_checkpoint(net, path, device):
    """
    Function to load model weights from a checkpoint.
    """
    if not isinstance(net, nn.Module):
        raise TypeError("The net parameter should be an instance of nn.Module.")
    net = net.to(device)
    print('==> Loading model from the checkpoint...')
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['state_dict'])
    return net
