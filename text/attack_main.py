from train_text import LanguageMIA
import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from utils import load_data
# fidn the current directory
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m-deduped')
    #parser.add_argument('--model_name', type=str, default='gpt2')
    #parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--shadow_num', type=int, default=30)
    parser.add_argument('--attack_size', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unlearn_method', type=str, default='ft')
    parser.add_argument('--save_path', type=str, default='../core/attack/attack_inferences/WikiText103')
    parser.add_argument('--prefix_epochs', type=int, default=1)
    parser.add_argument('--sft_epochs', type=int, default=10)
    parser.add_argument('--unlearn_epochs', type=int, default=3)


    args = parser.parse_args()


    if 'cuda' in args.device:
        device_idx = int(args.device.split(':')[1])
        torch.cuda.set_device(device_idx)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    args.tokenizer = tokenizer

    # Load preprocessed datasets
    print("[INFO] Loading datasets...")


    #target_dataset = load_from_disk('./core/data/WikiText-103-local/gpt2/random_dataset_prefixed')
    target_dataset = load_from_disk('./core/data/WikiText-103-local/pythia-70m/random_dataset_prefixed')







    #print random examples of target dataset
    # for i in range(16):
    #     print(tokenizer.decode(target_dataset[i]['input_ids'], skip_special_tokens=True))
    #     print("Attention Mask:", target_dataset[i]['attention_mask'])
    #     print("-" * 50)
    for i in range(16):
        input_ids = target_dataset[i]["input_ids"]
        last_7_ids = input_ids[-7:]

        decoded_tokens = tokenizer.convert_ids_to_tokens(last_7_ids)
        decoded_text = tokenizer.decode(last_7_ids)

        print(f"\n=== Sample {i} ===")
        print("Last 7 token IDs:", last_7_ids)
        print("Tokens:", decoded_tokens)
        print("Decoded:", decoded_text)



    # or any target dataset
    train_dataset, valid_dataset, _ = load_data("WikiText103", args)

    print(len(train_dataset))


    attack_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.attack_size))

    total_tokens = sum(len(sample['input_ids']) for sample in attack_dataset)
    print(f"Total number of tokens in the dataset: {total_tokens}")
    total_tokens_target = sum(len(sample['input_ids']) for sample in target_dataset)
    print(f"Total number of tokens in the target dataset: {total_tokens_target}")

    mia = LanguageMIA(target_dataset, valid_dataset, attack_dataset, tokenizer, args)

    # Run shadow models + inference
    print("[INFO] Starting shadow model training + inference...")
    results = mia.train_shadow_models()

    # print("results:", results)

    # Make sure the directory exists
    os.makedirs(args.save_path, exist_ok=True)
    filename = f"shadow_{args.shadow_num}_attack_random_{args.unlearn_method}_{args.model_name.replace('/', '_')}.pth"


    file_path = os.path.join(args.save_path, filename)

    # Ensure the directory exists
    os.makedirs(args.save_path, exist_ok=True)

    torch.save(results, file_path)
    torch.save(results, file_path)
    print(f"[INFO] Saved results to {args.save_path}")

    # # Summary (optional)
    # print("\n=== Summary ===")
    # for key in results:
    #     print(f"{key}: Collected for {len(results[key])} target samples")

if __name__ == '__main__':
    main()