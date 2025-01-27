import argparse
import torch
import random
import numpy as np
from torch import nn, optim
from evaluation.svc_mia import basic_mia, SVC_MIA, mia_threshold
from torch.optim.lr_scheduler import StepLR
from unlearn import (RandomLabel, GradientAscent, Retrain, Scrub, Fisher, BoundaryShrink, ScrubNew, FineTune, Certified,
                     SDD, GradientAscentPlus)
from utils import mul_loader, load_from_checkpoint, seed_everything
#from models import ResNet18, ResNet18_cifar100, vgg16_bn, wrn28_10
from utils.load_model import prepare_model, load_model, load_from_checkpoint
import time
from evaluation.accuracy import eval_accuracy
from evaluation.svc_mia import basic_mia, SVC_MIA, mia_threshold
from visual.inference import OutVisual


parser = argparse.ArgumentParser(description="Script for model unlearning with DUMP.")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument("--task", type=str, default="selective", help="Task to perform unlearning")
parser.add_argument("--forget_label", type=int, default=0, help='Forget label')
parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset to use for unlearning")
parser.add_argument("--trained_model_path", type=str, default=None, help="Path to the model checkpoint")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for unlearning")
parser.add_argument("--unlearn_epochs", type=int, default=10, help="Number of epochs for unlearning")
parser.add_argument("--forget_size", type=int, default=200, help="Number of samples to forget")
parser.add_argument('--upper_learning_rate', type=float, default=0.01)
parser.add_argument('--inner_learning_rate', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--scheduler_step_size', type=int, default=30, help='Step size for scheduler')
parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma for scheduler')
parser.add_argument('--forget_batch_size', type=int, default=50, help='Batch size for forget data')
parser.add_argument('--remain_batch_size', type=int, default=50, help='Batch size for remaining data')
parser.add_argument('--test_batch_size', type=int, default=128, help='Batch size for test data')
parser.add_argument('--arch', default='resnet18', help='Model architecture')
parser.add_argument('--unlearn_method', default='Retrain', help='Unlearning Method')
parser.add_argument('--num_workers', type=int, default=1, help='num workers for data loader')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for SGD')
parser.add_argument('--decreasing_lr', type=str, default="1, 3, 5,8", help='Epochs at which the learning rate should decrease')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory to save checkpoints')
parser.add_argument('--return_accuracy', action='store_true', help='Return accuracy after unlearning')
parser.add_argument('--return_mia_efficacy', action='store_true', help='Return Basic MIA after unlearning')
parser.add_argument('--plot_distributions', action='store_true', help='Plot distributions after unlearning')
#SCRUB ARGUMENTS
parser.add_argument('--m_steps', type=int, default=1, help='Number of steps for scrub')
parser.add_argument('--scrub_alpha', type=float, default=0.1, help='Alpha for scrub')
parser.add_argument('--scrub_beta', type=float, default=0.0, help='Beta for scrub')
parser.add_argument('--scrub_gamma', type=float, default=0.99, help='Gamma for scrub')
parser.add_argument('--T', type=int, default=1, help='Temperature for scrub')
parser.add_argument('--smoothing', type=float, default=0.1, help='Smoothing for scrub')
parser.add_argument('--lr_decay_epochs', type=str, default='1,3,5', help='Epochs at which to decay learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Rate of decay for learning rate')
parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer for scrub')
parser.add_argument('--with_l1', type=bool, default=False, help='Whether to use L1 regularization')
parser.add_argument('--sparse_alpha', type=float, default=0.001, help='Alpha for L1 regularization')
parser.add_argument('--no_l1_epochs', type=int, default=0, help='Number of epochs without L1 regularization')
parser.add_argument('--vulnerable_indices_path', type=str, default=None, help='Path to vulnerable indices')
parser.add_argument('--privacy_indices_path', type=str, default=None, help='Path to privacy indices')
parser.add_argument('--refine_epochs', type=int, default=5, help='Number of epochs for refining')
args = parser.parse_args()


def unlearn_main():

    print(f"Unlearning started Using {args.unlearn_method} Method on {args.dataset} Dataset with {args.task} Task")
    seed_everything(args.seed)

    if args.task == 'selective':
        print("Loading mixed data")
        mul_data = mul_loader.load_mul_data(args.dataset, args.task, args.forget_label, args.forget_size)
        forget_data = mul_data['forget']
        remain_data = mul_data['remain']

    if args.task == 'vulnerable or privacy':
        print("Loading mixed data")
        mul_data = mul_loader.load_mixed_data(args.dataset, args.vulnerable_indices_path, args.privacy_indices_path)


    mul_test_data = mul_loader.load_test_data(args.dataset, args.task)
    print("Forget data size: ", len(forget_data))
    print("Remain data size: ", len(remain_data))

    forget_loader = mul_loader._get_data_loader(forget_data, batch_size=args.forget_batch_size,
                                                num_workers=args.num_workers, shuffle=True)


    remain_loader = mul_loader._get_data_loader(remain_data, batch_size=args.remain_batch_size,
                                                num_workers=args.num_workers, shuffle=True)
    test_loader = mul_loader._get_data_loader(mul_test_data, batch_size=args.test_batch_size,
                                              num_workers=args.num_workers, shuffle=False)

    LOADER_DICT = {
        'forget': forget_loader,
        'remain': remain_loader,
        'test': test_loader
    }

    DATA_DICT = {
        'forget': forget_data,
        'remain': remain_data,
        'test': mul_test_data
    }

    model = prepare_model(args)
    print("Evaluating the model before unlearning")
    unlearn_utility(model, LOADER_DICT, DATA_DICT, args)
    unlearn_method_class = get_unlearning_method(args)
    visual_instance = OutVisual(model, LOADER_DICT['forget'], LOADER_DICT['remain'], LOADER_DICT['test'], args)
    visual_instance.plot_distributions('./pretrained')

    if args.unlearn_method == 'Retrain':

        unlearning_instance = unlearn_method_class(model, LOADER_DICT, args)
        start_time = time.time()
        unlearning_instance.unlearn()
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(model, LOADER_DICT, DATA_DICT, args)

    if args.unlearn_method == 'Scrub':
        model = load_from_checkpoint(model, args.trained_model_path)
        unlearning_instance = unlearn_method_class(model, LOADER_DICT, args, m_steps=1, scrub_alpha=0.1,
                                                   scrub_beta=0.0,
                                                   scrub_gamma=0.99, T=1)
        start_time = time.time()
        unlearned_model = unlearning_instance.unlearn()
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(unlearned_model, LOADER_DICT, DATA_DICT, args)

    if args.unlearn_method == 'GA':
        unlearning_instance = unlearn_method_class(model, LOADER_DICT, args)
        start_time = time.time()
        unlearned_model = unlearning_instance.unlearn()
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(unlearned_model, LOADER_DICT, DATA_DICT, args)

    if args.unlearn_method == 'Fisher':
        unlearning_instance = unlearn_method_class(model, LOADER_DICT, args, alpha=1e-6)
        start_time = time.time()
        unlearned_model = unlearning_instance.unlearn()
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(unlearned_model, LOADER_DICT, DATA_DICT, args)

    if args.unlearn_method == 'BS':
        unlearning_instance = unlearn_method_class(model, LOADER_DICT, args)
        start_time = time.time()
        unlearning_instance.unlearn()
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(model, LOADER_DICT, DATA_DICT, args)

    if args.unlearn_method == 'RL':
        unlearning_instance = unlearn_method_class(model, LOADER_DICT, args)
        start_time = time.time()
        unlearning_instance.unlearn()
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(model, LOADER_DICT, DATA_DICT, args)

    if args.unlearn_method == 'ScrubNew':
        unlearning_instance = unlearn_method_class(model, LOADER_DICT, args)
        start_time = time.time()
        unlearned_model = unlearning_instance.unlearn()
        print(model.parameters())
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(unlearned_model, LOADER_DICT, DATA_DICT, args)

    if args.unlearn_method == 'FT':
        unlearning_instance = unlearn_method_class(model, LOADER_DICT, args)
        start_time = time.time()
        unlearned_model = unlearning_instance.unlearn()
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(unlearned_model, LOADER_DICT, DATA_DICT, args)

    if args.unlearn_method == 'CE':
        unlearn_utility(model, LOADER_DICT, DATA_DICT, args)
        unlearning_instance = unlearn_method_class(model, LOADER_DICT, DATA_DICT, args)
        start_time = time.time()
        unlearned_model = unlearning_instance.unlearn()
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(unlearned_model, LOADER_DICT, DATA_DICT, args)

    if args.unlearn_method == 'GA+':
        unlearning_instance = GradientAscentPlus(model, LOADER_DICT, args)
        start_time = time.time()
        unlearned_model = unlearning_instance.unlearn()
        end_time = time.time()
        print(f"Time taken for unlearning: {end_time - start_time} seconds")
        unlearn_utility(unlearned_model, LOADER_DICT, DATA_DICT, args)


    # if args.unlearn_method == 'SDD':
    #     unlearn_utility(model, LOADER_DICT, DATA_DICT, args)
    #     unlearning_instance = SDD(model, LOADER_DICT, DATA_DICT, args)
    #     print("Start Unlearning")
    #     start_time = time.time()
    #     unlearning_instance.unlearn()
    #     end_time = time.time()
    #     print(f"Time taken for unlearning: {end_time - start_time} seconds")
        #unlearn_utility(unlearned_model, LOADER_DICT, DATA_DICT, args

    #torch.save(model.state_dict(), f'./pretrained/{args.unlearn_method}_{args.dataset}_{args.task}_{args.seed}_'
                                   #f'{args.forget_size}.pth')


def get_unlearning_method(args):
    unlearning_methods = {
        'RL': RandomLabel,
        'GA': GradientAscent,
        'Retrain': Retrain,
        'Scrub': Scrub,
        'SDD': SDD,
        'Fisher': Fisher,
        'BS': BoundaryShrink,
        'ScrubNew': ScrubNew,
        'FT': FineTune,
        'CE': Certified,
        'GA+': GradientAscentPlus

        #TODO
        # 'SCRUB': Scrub,
        # 'FT': FineTuning,
        # 'FI': FisherInformation,
    }

    if args.unlearn_method not in unlearning_methods:
        raise ValueError('Invalid unlearn method')

    unlearn_method_class = unlearning_methods[args.unlearn_method]
    return unlearn_method_class


def unlearn_utility(model, LOADER_DICT, DATA_DICT, args):
    if args.return_accuracy:
        print("Evaluating the model after unlearning")
        print("ACC on forget data", eval_accuracy(model, LOADER_DICT['forget'], args.device))
        print("ACC on remain data", eval_accuracy(model, LOADER_DICT['remain'], args.device))
        print("ACC on test data", eval_accuracy(model, LOADER_DICT['test'], args.device))

    if args.return_mia_efficacy:
        print("Evaluating the MIA efficacy after unlearning")
        mia_efficacy_results = basic_mia(model, DATA_DICT['forget'], DATA_DICT['remain'], DATA_DICT['test'])
        mia_efficacy_forget = mia_efficacy_results['forget']
        mia_efficacy_remain = mia_efficacy_results['remain']
        print("MIA Efficacy Results")
        print("MIA on forget data", mia_efficacy_forget)
        print("MIA on remain data", mia_efficacy_remain)

    if args.plot_distributions:

        print("Plotting distributions after unlearning")
        visual_instance = OutVisual(model, LOADER_DICT['forget'], LOADER_DICT['remain'], LOADER_DICT['test'], args)
        visual_instance.plot_distributions(args.checkpoint_dir)

    else:
        pass


if __name__ == "__main__":
    unlearn_main()




