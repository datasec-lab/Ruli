from train_text import LanguageMIA
import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from utils import load_data
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m-deduped')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--shadow_num', type=int, default=30)
    parser.add_argument('--attack_size', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--unlearn_method', type=str, default='ft')
    parser.add_argument('--save_path', type=str, default='../core/attack/attack_inferences/WikiText103')
    parser.add_argument('--prefix_epochs', type=int, default=1)
    parser.add_argument('--sft_epochs', type=int, default=10)
    parser.add_argument('--unlearn_epochs', type=int, default=3)
    parser.add_argument('--target_data_path', type=str, default='./data/WikiText-103-local/gpt2/random_dataset_prefixed')


    args = parser.parse_args()


    if 'cuda' in args.device:
        device_idx = int(args.device.split(':')[1])
        torch.cuda.set_device(device_idx)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    args.tokenizer = tokenizer

    # Load preprocessed datasets
    print("[INFO] Loading datasets...")
    target_dataset = load_from_disk(args.target_data_path)
    train_dataset, valid_dataset, _ = load_data("WikiText103", args)
    attack_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.attack_size))
    total_tokens = sum(len(sample['input_ids']) for sample in attack_dataset)
    print(f"Total number of tokens in the dataset: {total_tokens}")
    total_tokens_target = sum(len(sample['input_ids']) for sample in target_dataset)
    print(f"Total number of tokens in the target dataset: {total_tokens_target}")


    mia = LanguageMIA(target_dataset, valid_dataset, attack_dataset, tokenizer, args)
    print("[INFO] Starting shadow model training + inference...")
    results = mia.train_shadow_models()


    os.makedirs(args.save_path, exist_ok=True)
    filename = f"shadow_{args.shadow_num}_attack_random_{args.unlearn_method}_{args.model_name.replace('/', '_')}.pth"
    file_path = os.path.join(args.save_path, filename)
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(results, file_path)
    print(f"[INFO] Saved results to {args.save_path}")


if __name__ == '__main__':
    main()