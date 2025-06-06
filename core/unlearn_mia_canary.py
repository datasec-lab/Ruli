import os
import argparse
import torch
import copy
import torch.utils.data as data
from attack.unlearn_attack import MUlMIA, TestEvaluator
from attack.unlearn_attack_canary import CanaryEvaluator
from attack.train import Retrain
from utils.loader import mul_loader
from utils.seed import seed_everything
from utils import prepare_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='MIA Threshold')
    parser.add_argument('--unlearn_method', default='Scrub', type=str, help='Unlearning method')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset')
    parser.add_argument('--task', default='selective', type=str, help='Task')
    parser.add_argument('--forget_size', default=600, type=int, help='Forget size')
    parser.add_argument('--output_type', default='logit-scaled-confidences', type=str, help='Output type')
    parser.add_argument('--device', default='cuda:1', type=str, help='Device')
    parser.add_argument('--trained_model_path', default=None, type=str, help='Trained model path')
    parser.add_argument('--arch', default='resnet18',
                        choices=['resnet18', 'resnet50', 'vit_tiny', 'vgg16_bn', 'wrn28_10', 'vit', 'resnet_tin',
                                 'resnext50', 'resnet18_TinyImageNet'],
                        type=str, help='Architecture')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for attack data')
    parser.add_argument('--attack_size', default=25000, type=int, help='Attack size')
    parser.add_argument('--train_epochs', default=50, type=int, help='Number of epochs for retraining')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    # parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--weight_decay', default=5e-5, type=float, help='Weight decay')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/cifar10', type=str, help='Checkpoint directory')
    parser.add_argument('--forget_label', default=0, type=int, help='Forget label')
    parser.add_argument('--shadow_num', default=90, type=int, help='Number of shadow models')
    parser.add_argument('--example_index', nargs='+', default=[5], type=int,
                        help='List of example image indices for monitoring')
    parser.add_argument('--parallel', default=False, type=bool, help='Parallel training')
    parser.add_argument('--result_path', default='./attack/attack_inferences', type=str, help='Result path')
    parser.add_argument('--train_shadow', action='store_true', help='Train shadow models')
    parser.add_argument('--saved_results', default=None, type=str, help='Path to saved results')
    parser.add_argument('--target_model_path', default='./pretrained/resnet18/cifar10.pth', type=str,
                        help='Path to target model')
    parser.add_argument('--vulnerable_path', default='./attack/cifar10', type=str, help='Path to vulnerable data')
    parser.add_argument('--privacy_path', default='./data/cifar10', type=str, help='Path to file data')
    parser.add_argument('--return_accuracy', action='store_true', help='Return accuracy after unlearning')
    parser.add_argument('--plot_distributions', action='store_true', help='Plot distributions after unlearning')
    parser.add_argument('--forget_batch_size', default=16, type=int, help='Forget batch size')
    parser.add_argument('--remain_batch_size', default=64, type=int, help='Forget epochs')
    parser.add_argument('--test_batch_size', default=128, type=int, help='Forget epochs')
    parser.add_argument('--config_path', default='./unlearn_config.json', type=str,
                        help="this the config file path for the unlearning parameters")
    return parser.parse_args()



def split_indices(args, target_data):
    """Split indices into exclusive subsets for canary or general tasks."""
    torch.manual_seed(args.seed)  # Fix the seed for reproducibility
    total_indices = torch.randperm(len(target_data), generator=torch.Generator().manual_seed(args.seed)).tolist()
    total_indices = sorted(total_indices)

    if args.task == 'canary':
        # Filter indices below 600
        filtered_indices = [idx for idx in total_indices if idx < 600]
        assert len(filtered_indices) >= 600, f"Not enough indices below 600, found {len(filtered_indices)}"

        # Shuffle filtered indices
        filtered_indices_tensor = torch.tensor(filtered_indices)
        shuffled_filtered = filtered_indices_tensor[torch.randperm(len(filtered_indices_tensor))]

        # Slice subsets
        in_data_canary_indices = shuffled_filtered[:100].tolist()
        unlearned_data_canary_indices = shuffled_filtered[100:350].tolist()
        out_data_canary_indices = shuffled_filtered[350:600].tolist()

        # Convert to tensors
        in_data_indices_ = torch.tensor(in_data_canary_indices)
        out_data_indices_ = torch.tensor(out_data_canary_indices)
        unlearned_data_indices_ = torch.tensor(unlearned_data_canary_indices)

        # Exclude used indices from total
        used_indices = set(in_data_canary_indices + unlearned_data_canary_indices + out_data_canary_indices)
        remaining_after_unlearn = list(set(total_indices) - used_indices)

        # Shuffle remaining indices
        remaining_after_unlearn_tensor = torch.tensor(remaining_after_unlearn)
        shuffled_remaining = remaining_after_unlearn_tensor[torch.randperm(len(remaining_after_unlearn_tensor))]

        # Split remaining indices
        split_point_above_600 = len(shuffled_remaining) // 3
        in_data_canary_indices_above = shuffled_remaining[:split_point_above_600].tolist()
        out_data_canary_indices_above = shuffled_remaining[split_point_above_600:2 * split_point_above_600].tolist()
        unlearned_data_canary_indices_above = shuffled_remaining[2 * split_point_above_600:].tolist()

        # Merge subsets
        in_data_indices = torch.cat((in_data_indices_, torch.tensor(in_data_canary_indices_above)))
        out_data_indices = torch.cat((out_data_indices_, torch.tensor(out_data_canary_indices_above)))
        unlearned_data_indices = torch.cat((unlearned_data_indices_, torch.tensor(unlearned_data_canary_indices_above)))

    else:
        # General task: shuffle and split indices
        shuffled_indices = torch.tensor(total_indices)[torch.randperm(len(total_indices))]
        split_point = len(shuffled_indices) // 3
        in_data_indices = shuffled_indices[:split_point]
        out_data_indices = shuffled_indices[split_point:2 * split_point]
        unlearned_data_indices = shuffled_indices[2 * split_point:]

    return in_data_indices, out_data_indices, unlearned_data_indices


def canary_injection(args):
    """Perform canary injection and unlearning."""
    seed_everything(args.seed)

    # Load data
    split_data = mul_loader.load_mixed_vulnerable_data(args.dataset, args.vulnerable_path, args.privacy_path)
    target_data, remain_data = split_data['forget'], split_data['remain']
    train_data, test_data = mul_loader.load_data(args.dataset)
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize attack instance
    attack_instance = MUlMIA(target_data, test_data, remain_data, args)
    shadow_results = attack_instance.run_attack() if args.train_shadow else torch.load(args.saved_results, weights_only=False)

    # Prepare model
    target_model = prepare_model(args, fresh_seed=True)

    # Split indices
    in_data_indices, out_data_indices, unlearned_data_indices = split_indices(args, target_data)

    # Create subsets
    in_data = data.Subset(target_data, in_data_indices)
    out_data = data.Subset(target_data, out_data_indices)
    unlearned_data = data.Subset(target_data, unlearned_data_indices)
    remain_indices = torch.randperm(len(remain_data))[:args.attack_size]
    remain_data = data.Subset(remain_data, remain_indices)

    # Combine datasets
    train_data = data.ConcatDataset([in_data, remain_data, unlearned_data])
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Unlearning
    loader_dict = {'train': train_loader, 'test': test_loader}
    unlearning_instance = Retrain(target_model, loader_dict, args)
    unlearning_instance.unlearn()

    # Evaluate
    trained_model = copy.deepcopy(target_model)
    method_config = MUlMIA.load_unlearn_config(args.config_path)[args.unlearn_method]
    args.forget_batch_size = method_config.get('forget_batch_size')
    args.remain_batch_size = method_config.get('remain_batch_size')

    forget_loader = data.DataLoader(unlearned_data, batch_size=args.forget_batch_size, shuffle=True, num_workers=args.num_workers)
    remain_loader = data.DataLoader(remain_data, batch_size=args.remain_batch_size, shuffle=True, num_workers=args.num_workers)
    unlearn_loader = {'forget': forget_loader, 'remain': remain_loader, 'test': test_loader}

    unlearned_model = attack_instance.unlearn_shadow_model(
        model=target_model, forget_dict=unlearn_loader,
        forget_data={'forget': unlearned_data, 'remain': remain_data, 'test': test_data},
        method=args.unlearn_method
    )

    MUlMIA.unlearn_utility(unlearned_model, unlearn_loader, {'forget': unlearned_data, 'remain': remain_data, 'test': test_data}, args)
    evaluator = CanaryEvaluator(trained_model, unlearned_model, target_data, shadow_results, in_data_indices, unlearned_data_indices, out_data_indices, args)
    evaluator.run_evaluation()


def main():
    args = parse_args()
    canary_injection(args)


if __name__ == "__main__":
    main()