
import argparse
import torch
import copy
import torch.utils.data as data
from attack.unlearn_attack import MUlMIA, TargetModelEvaluator, TestEvaluator
from attack.train import Retrain
from utils.loader import mul_loader
from utils.seed import seed_everything
from utils import prepare_model



def parse_args():
    parser = argparse.ArgumentParser(description='MIA Threshold')
    parser.add_argument('--unlearn_method', default='ScrubNew', type=str, help='Unlearning method')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset')
    parser.add_argument('--task', default='selective', type=str, help='Task')
    parser.add_argument('--forget_size', default=600, type=int, help='Forget size')
    parser.add_argument('--output_type', default='logit', type=str, help='Output type')
    parser.add_argument('--device', default='cuda:1', type=str, help='Device')
    parser.add_argument('--trained_model_path', default=None, type=str, help='Trained model path')
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'vgg16_bn', 'wrn28_10', 'vit'],
                        type=str, help='Architecture')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for attack data')
    parser.add_argument('--attack_size', default=24000, type=int, help='Attack size')
    parser.add_argument('--train_epochs', default=50, type=int, help='Number of epochs for retraining')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/cifar10', type=str, help='Checkpoint directory')
    parser.add_argument('--forget_label', default=0, type=int, help='Forget label')
    parser.add_argument('--shadow_num', default=36, type=int, help='Number of shadow models')
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
    parser.add_argument('--return_mia_efficacy', action='store_true', help='Return Basic MIA after unlearning')
    parser.add_argument('--plot_distributions', action='store_true', help='Plot distributions after unlearning')
    parser.add_argument('--forget_batch_size', default=16, type=int, help='Forget batch size')
    parser.add_argument('--remain_batch_size', default=64, type=int, help='Forget epochs')
    parser.add_argument('--test_batch_size', default=128, type=int, help='Forget epochs')
    return parser.parse_args()



def ruli_attack(args):
    seed_everything(args.seed)
    #print(args.seed)
    target_data, remain_data = None, None
    if args.task == 'selective' or args.task == 'class-wise':
        split_data = mul_loader.load_mul_data(args.dataset, args.task, f_label=5, forget_size=args.forget_size)
        target_data, remain_data = split_data['forget'], split_data['remain']

    if args.task == 'mixed':
        #split_data = mul_loader.load_mixed_data(args.dataset, args.vulnerable_path, args.privacy_path)
        split_data = mul_loader.load_mixed_data(args.dataset, args.vulnerable_path, args.privacy_path)
        target_data, remain_data = split_data['forget'], split_data['remain']

    if args.task == 'vulnerable':
        split_data = mul_loader.load_mixed_data(args.dataset, args.vulnerable_path, args.privacy_path)
        target_data, remain_data = split_data['vulnerable'], split_data['remain']

    if args.task == 'canary':
        split_data = mul_loader.load_mixed_vulnerable_data(args.dataset, args.vulnerable_path, args.privacy_path)
        target_data, remain_data = split_data['forget'], split_data['remain']

    if args.task == 'privacy':
        split_data = mul_loader.load_mixed_data(args.dataset, args.vulnerable_path, args.privacy_path)
        target_data, remain_data = split_data['privacy'], split_data['remain']
        print("Number of samples in target data")
        print(len(split_data['vulnerable_index']))
        print(len(split_data['privacy_index']))

    #print(split_data['forget_index'])

    train_data, test_data = mul_loader.load_data(args.dataset)
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
    attack_instance = MUlMIA(target_data, test_data, remain_data, args)
    if args.train_shadow:
        shadow_results = attack_instance.run_attack()
    else:
        with open(args.saved_results, 'rb') as f:
            shadow_results = torch.load(f)


    target_model = prepare_model(args, fresh_seed=True)
    torch.manual_seed(args.seed)
    total_indices = torch.randperm(len(target_data))
    split_point = len(target_data) // 3
    in_data_indices, out_data_indices, unlearned_data_indices = (total_indices[:split_point],
                                                                 total_indices[split_point:2*split_point], total_indices[2*split_point:])


    in_data = data.Subset(target_data, in_data_indices)
    out_data = data.Subset(target_data, out_data_indices)
    unlearned_data = data.Subset(target_data, unlearned_data_indices)

    remain_indices = torch.randperm(len(remain_data))[:args.attack_size]
    remain_data = data.Subset(remain_data, remain_indices)
    train_data = data.ConcatDataset([in_data, remain_data])
    train_data = data.ConcatDataset([train_data, unlearned_data])

    print(f"Number of train data: {len(train_data)}")
    print(f"Number of in data: {len(in_data)}")
    print(f"Number of out data: {len(out_data)}")
    print(f"Number of unlearned data: {len(unlearned_data)}")

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    loader_dict = {'train': train_loader, 'test': test_loader}
    in_data_indices, out_data_indices = in_data_indices.tolist(), out_data_indices.tolist()
    overlapping_in_out = set(in_data_indices).intersection(set(out_data_indices))
    overlapping_in_unlearned = set(in_data_indices).intersection(set(unlearned_data_indices))
    print(f"Number of overlapping samples in in/out: {len(overlapping_in_out)}")
    print(f"Number of overlapping samples in in/unlearned: {len(overlapping_in_unlearned)}")
    unlearning_instance = Retrain(target_model, loader_dict, args)
    unlearning_instance.unlearn()
    trained_model = copy.deepcopy(target_model)


    forget_dict_eval = {'forget': data.DataLoader(train_data, batch_size=args.forget_batch_size, shuffle=True,
                                                  num_workers=args.num_workers),
                        'remain': data.DataLoader(remain_data, batch_size=args.remain_batch_size, shuffle=True,
                                                  num_workers=args.num_workers),
                        'test': data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True,
                                                num_workers=args.num_workers)}
    MUlMIA.unlearn_utility(trained_model, forget_dict_eval,
                           {'forget': train_data, 'remain': remain_data, 'test': test_data}, args)
    forget_loader = data.DataLoader(unlearned_data, batch_size=args.forget_batch_size, shuffle=True,
                                    num_workers=args.num_workers)

    remain_data = data.ConcatDataset([remain_data, in_data])
    remain_loader = data.DataLoader(remain_data, batch_size=args.remain_batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    test_loader = data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True,
                                  num_workers=args.num_workers)

    unlearn_loader = {'forget': forget_loader, 'remain': remain_loader, 'test': test_loader}
    unlearned_model = attack_instance.unlearn_shadow_model(model=target_model, forget_dict=unlearn_loader,
                                                           forget_data={'forget': unlearned_data,
                                                                        'remain': remain_data,
                                                                        'test': test_data},
                                                           method=args.unlearn_method)

    MUlMIA.unlearn_utility(unlearned_model, unlearn_loader,
                           {'forget': unlearned_data, 'remain': remain_data, 'test': test_data}, args)

    efficay = TestEvaluator(trained_model, unlearned_model, trained_model, target_data, shadow_results,
                             in_data_indices, unlearned_data_indices, out_data_indices, args)

    privacy_leakage = TargetModelEvaluator(trained_model, unlearned_model, target_data, shadow_results,
                             in_data_indices, unlearned_data_indices, out_data_indices, args)
    efficay.run_evaluation()
    privacy_leakage.run_evaluation()
    privacy_leakage.run_population_attack()
    privacy_leakage.run_population_attack_vulnerable()



if __name__ == '__main__':
    args = parse_args()
    ruli_attack(args)