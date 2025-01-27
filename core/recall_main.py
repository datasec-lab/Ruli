
from utils.loader import mul_loader
from attack.train import Retrain
from utils import prepare_model, seed_everything
import argparse
from attack.memory_attack import MIA, TargetModelEvaluator  # Assuming the MIA class is in a file called mia_attack.py
import torch.utils.data as data
import torch
import numpy as np
from sklearn.metrics import accuracy_score


def parse_args():

    parser = argparse.ArgumentParser(description='MIA Threshold')
    parser.add_argument('--dataset', default='cifar100', type=str, help='dataset')
    parser.add_argument('--task', default='selective', type=str, help='task')
    parser.add_argument('--forget_size', default=3000,
                        type=int, help='forget size')
    parser.add_argument('--output_type', default='logit', type=str, help='output type')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--trained_model_path', default=None, type=str, help='trained model path')
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'vgg16_bn', 'wrn28_10', 'vit', 'resnext50'], type=str, help='architecture')
    parser.add_argument('--seed', default=0, type=int, help='random seed for attack data')
    parser.add_argument('--attack_size', default=25000, type=int, help='attack size')
    parser.add_argument('--train_epochs', default=50, type=int, help='number of epochs for retraining')
    parser.add_argument('--lr', default=0.1, type=float, help='zlearning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for dataloader')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/cifar10', type=str, help='checkpoint directory')
    parser.add_argument('--forget_label', default=0, type=int, help='forget labsel')
    parser.add_argument('--shadow_num', default=64, type=int, help='number of shadow models')
    parser.add_argument('--example_index', nargs='+', default=[450], type=int,
                        help='List of example image indices for monitoring')
    parser.add_argument('--parallel', default=False, type=bool, help='parallel training')
    parser.add_argument('--result_path', default='./attack/attack_inferences', type=str, help='result path')
    parser.add_argument('--train_shadow', default=True, type=bool, help='train shadow models')
    parser.add_argument('--saved_results', default=None, type=str, help='path to saved results')
    parser.add_argument('--target_model_path', default='./pretrained/resnet18/cifar10.pth', type=str,
                        help='path to target model')
    return parser.parse_args()


def main():
    args = parse_args()
    train_data, target_data, shadow_results = load_data_shadow(args)
    _, test_data = mul_loader.load_data(args.dataset)
    target_model, evaluator, target_model_switch, evaluator_switch = setup_model_and_evaluator(args, target_data,
                                                                                     test_data, shadow_results)
    sample_likelihoods, true_labels, likelihood_ratios = evaluator.evaluate_sample_likelihood()
    fpr, tpr, thresholds, roc_auc = evaluator.calculate_roc(true_labels, likelihood_ratios)
    print(fpr)
    print(tpr)
    report_fpr_threshold = 0.001# 0.1%
    low_fpr_threshold = 0.0001   # 0.01%
    high_fpr_threshold = 0.9
    tpr_at_report_fpr, _ = find_tpr_at_fpr(fpr, tpr, thresholds, report_fpr_threshold)
    tpr_at_001_fpr, _ = find_tpr_at_fpr(fpr, tpr, thresholds, 0.001)
    tpr_at_0001_fpr, _ = find_tpr_at_fpr(fpr, tpr, thresholds, 0.0001)
    tpr_at_00001_fpr, _ = find_tpr_at_fpr(fpr, tpr, thresholds, 0.00001)
    tpr_at_000001_fpr, _ = find_tpr_at_fpr(fpr, tpr, thresholds, 0.000001) # 0.0001%
    tpr_at_00003_fpr, _ = find_tpr_at_fpr(fpr, tpr, thresholds, 0.0003)
    tpr_at_01_fpr, _ = find_tpr_at_fpr(fpr, tpr, thresholds, 0.01)
    print(f"TPR at FPR 1%: {tpr_at_report_fpr}") # 1%
    print(f"TPR at FPR 0.1%: {tpr_at_001_fpr}")
    print(f"TPR at FPR 0.01%: {tpr_at_0001_fpr}")
    print(f"TPR at FPR 0.001%: {tpr_at_00001_fpr}")
    print(f"TPR at FPR 0.0001%: {tpr_at_000001_fpr}")
    predicted_labels = (np.array(likelihood_ratios) > 0.5).astype(int)
    attack_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Attack Accuracy based on raw likelihoods: {attack_accuracy:.4f}")

    target_sample_count = 5
    vulnerable_samples, fpr_at_threshold, tpr_at_threshold = evaluator.find_fpr_for_sample_count(
        true_labels, likelihood_ratios, target_sample_count
    )

    print(f"Number of vulnerable samples: {len(vulnerable_samples)}")
    print(f"TPR at threshold: {tpr_at_threshold}")
    print(f"FPR at threshold: {fpr_at_threshold}")
    print(f"Threshold: {thresholds[np.argmin(np.abs(fpr - fpr_at_threshold))]}")

    vulnerable_samples, threshold_at_fpr, tpr_at_fpr = evaluator.find_vulnerable_samples_at_fpr(true_labels,
                                                                                                likelihood_ratios,
                                                                                                low_fpr_threshold)
    print("vulnerable samples: ", vulnerable_samples)
    non_vulnerable_samples, threshold_high, tpr_high = evaluator.find_non_vulnerable_samples_at_fpr(true_labels, likelihood_ratios,
                                                                            high_fpr_threshold)


    privacy_samples, threshold_at_fpr, tpr_at_fpr = evaluator.find_privacy_preserving_samples(true_labels,
                                                                                                likelihood_ratios,
                                                                                                report_fpr_threshold,
                                                                                                tolerance=0.1)

    log_results(roc_auc, tpr_at_report_fpr, low_fpr_threshold, tpr_at_fpr, len(vulnerable_samples),
                len(vulnerable_samples), args)
    log_results(roc_auc, tpr_at_report_fpr, high_fpr_threshold, tpr_high, len(non_vulnerable_samples),
                len(non_vulnerable_samples), args)


def load_data_shadow(args):
    train_data, test_data = mul_loader.load_data(args.dataset)
    target_data = train_data  # Use the training data for MIA

    attack_instance = MIA(target_data, train_data, test_data, args)
    if args.train_shadow is True:
        shadow_results = attack_instance.run_attack()
    else:
        with open(args.saved_results, 'rb') as f:
            shadow_results = torch.load(args.saved_results)

    return train_data, target_data, shadow_results


def setup_model_and_evaluator(args, target_data, test_data, shadow_results):
    shadow_logits_in = shadow_results['in']
    shadow_logits_out = shadow_results['out']
    target_model = prepare_model(args, fresh_seed=False)
    seed_everything(args.seed)
    in_data_indices = torch.randperm(len(target_data))[:len(target_data) // 2]
    print(in_data_indices[:10])
    in_data = data.Subset(target_data, in_data_indices)
    out_data_indices = torch.randperm(len(target_data))[len(target_data) // 2:]
    out_data = data.Subset(target_data, out_data_indices)
    loader_dict = {
        'train': data.DataLoader(in_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        'test': data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }

    loader_dic_switch = {
        'train': data.DataLoader(out_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        'test': data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }

    unlearning_instance = Retrain(target_model, loader_dict, args)
    unlearning_instance.unlearn()
    evaluator = TargetModelEvaluator(target_model, target_data, shadow_logits_in, shadow_logits_out, in_data_indices,
                                     args)

    target_model_switch = prepare_model(args, fresh_seed=False)
    unlearning_instance_switch = Retrain(target_model_switch, loader_dic_switch, args)
    unlearning_instance_switch.unlearn()
    evaluator_switch = TargetModelEvaluator(target_model_switch, target_data, shadow_logits_in, shadow_logits_out, out_data_indices,
                                     args)

    return target_model, evaluator, target_model_switch, evaluator_switch


def find_tpr_at_fpr(fpr, tpr, thresholds, desired_fpr):
    low_fpr_index = np.argmin(np.abs(fpr - desired_fpr))
    threshold_at_fpr = thresholds[low_fpr_index]
    tpr_at_fpr = tpr[low_fpr_index]
    return tpr_at_fpr, threshold_at_fpr


def log_results(roc_auc, tpr_at_report_fpr, fpr_at_threshold, tpr_at_threshold, num_memorized_samples,
                num_non_memorized_samples, args):
    print("\n" + "=" * 60)
    print(f"{'Starting Memory Inference Attack':^60}")
    print("=" * 60)
    print(f"{'Dataset:':<20} {args.dataset}")
    print(f"{'Seed:':<20} {args.seed}")
    print(f"{'Attack Size:':<20} {args.attack_size}")
    print(f"{'Shadow Models:':<20} {args.shadow_num}")
    print(f"{'Output Type:':<20} {args.output_type}")
    print(f"{'Model Architecture:':<20} {args.arch}")
    print(f"{'ROC AUC:':<20} {roc_auc:.4f}")
    print(f"{'TPR @ FPR 0.001:':<20} {tpr_at_report_fpr:.4f}")
    print(f"{'FPR at Threshold:':<20} {fpr_at_threshold:.4f}")
    print(f"{'TPR at Threshold:':<20} {tpr_at_threshold:.4f}")
    print(f"{'Number of Vulnerable Samples:':<20} {num_memorized_samples}")
    print(f"{'Number of Non-Vulnerable Samples:':<20} {num_non_memorized_samples}")
    print("=" * 60)

    result_file_name = f"results_{args.dataset}_{args.seed}_{args.shadow_num}.txt"
    with open(result_file_name, 'w') as f:
        f.write("\n" + "=" * 60)
        f.write(f"{'Starting Memory Inference Attack':^60}")
        f.write("=" * 60)
        f.write(f"{'Dataset:':<20} {args.dataset}")
        f.write(f"{'Seed:':<20} {args.seed}")
        f.write(f"{'Attack Size:':<20} {args.attack_size}")
        f.write(f"{'Shadow Models:':<20} {args.shadow_num}")
        f.write(f"{'Output Type:':<20} {args.output_type}")
        f.write(f"{'Model Architecture:':<20} {args.arch}")
        f.write(f"{'ROC AUC:':<20} {roc_auc:.4f}")
        f.write(f"{'TPR @ FPR 0.01:':<20} {tpr_at_report_fpr:.4f}")
        f.write(f"{'FPR at Threshold:':<20} {fpr_at_threshold:.4f}")
        f.write(f"{'TPR at Threshold:':<20} {tpr_at_threshold:.4f}")
        f.write(f"{'Number of Vulnerable Samples:':<20} {num_memorized_samples}")
        f.write("=" * 60)


if __name__ == '__main__':
    main()




