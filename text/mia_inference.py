import argparse
import random
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Subset, ConcatDataset
from utils import MIAEvaluator, EfficacyEvaluator
from utils import train_sft, unlearn_model, train_prefix
from utils import load_data
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m-deduped')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--shadow_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--sft_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unlearn_method', type=str, default='npo')
    parser.add_argument('--unlearn_epochs', type=int, default=15)
    parser.add_argument('--target_data_path', type=str)
    parser.add_argument('--attack_size', type=int, default=15000)
    parser.add_argument('--prefix_epochs', type=int, default=1)
    parser.add_argument('--FT_epochs', type=int, default=2)



    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading datasets...")
    target_dataset = load_from_disk(args.target_data_path)

    train_dataset, valid_dataset, _ = load_data("WikiText103", args)
    attack_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.attack_size))


    print("[INFO] Loading shadow results...")
    shadow_results = torch.load(args.shadow_path)

    total_indices = sorted(list(shadow_results['in_original'].keys()))
    assert len(total_indices) >= 600

    in_ids = total_indices[:200]
    unlearn_ids = total_indices[200:400]
    out_ids = total_indices[400:600]


    ##################################
    in_data = Subset(target_dataset, in_ids)
    unlearn_data = Subset(target_dataset, unlearn_ids)
    out_data = Subset(target_dataset, out_ids)


    ########################


    train_data = ConcatDataset([in_data, unlearn_data, attack_dataset])
    retain_dataset = ConcatDataset([in_data, attack_dataset])

    print("[INFO] Training model on IN + UNLEARN + attack...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    train_sft(model, train_data, valid_dataset, tokenizer, args.sft_epochs)
    train_prefix(model, train_data, valid_dataset, tokenizer, args.prefix_epochs)
    original_model = copy.deepcopy(model)

    unlearned_model = unlearn_model(model, unlearn_data, retain_dataset, valid_dataset, tokenizer, args)
    train_sft(unlearned_model, retain_dataset, valid_dataset, tokenizer, args.FT_epochs)

    # retraining
    # unlearned_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    # train_sft(unlearned_model, retain_dataset, valid_dataset, tokenizer, args.sft_epochs)
    # train_prefix(unlearned_model, retain_dataset, valid_dataset, tokenizer, args.prefix_epochs)


    print("[INFO] Running MIA evaluation...")
    evaluator = MIAEvaluator(
        target_model=original_model,
        unlearned_model=unlearned_model,
        target_dataset=target_dataset,
        tokenizer=tokenizer,
        device=device,
        args=args
    )

    results = evaluator.run(
        shadow_results=shadow_results,
        out_ids=out_ids,
        unlearn_ids=unlearn_ids
    )

    print("\n=== MIA Evaluation Metrics ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("[INFO] MIA evaluation completed.")
    print("\n=== Efficacy Evaluation Results ===")

    evaluator = EfficacyEvaluator(
        target_model=original_model,
        unlearned_model=unlearned_model,
        target_dataset=target_dataset,
        tokenizer=tokenizer,
        device=device,
        args=args
    )

    results = evaluator.run(
        shadow_results=shadow_results,
        out_ids=out_ids,
        unlearn_ids=unlearn_ids
    )
    print("\n=== Efficacy Evaluation Metrics ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")




if __name__ == '__main__':
    main()
