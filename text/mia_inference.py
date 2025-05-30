import argparse
import random
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Subset, ConcatDataset
from utils import MIAEvaluator, EfficacyEvaluator
from utils import train_sft, unlearn_model, train_prefix, unlearn_prefix, summarize_out_unlearn_7gram_overlap, summarize_out_unlearn_first7gram_overlap
from utils import load_data
import copy
from utils import (analyze_7gram_overlap_tokenized, select_high_overlap_out_unl_ids_global, compare_first_final_7gram_overlap,
                   compare_7gram_overlap_modes, run_blind_overlap_attack, analyze_7gram_overlap_and_shift, compute_out_overlap_with_target)
from utils import classification_report, run_bow_mia_baseline, run_bow_on_last_tokens
from utils import compute_final_7gram_overlap_in_validation, compute_token_and_1gram_overlap



def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m-deduped')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--shadow_path', type=str, required=True, default= '../core/attack/attack_inferences/WikiText103/PREFIX_PREFIX_shadow_15_attack_random_ga+gdr_EleutherAI_pythia-70m-deduped.pth')
    # parser.add_argument('--attack_data_path', type=str, required=True)
    # parser.add_argument('--valid_data_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--sft_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unlearn_method', type=str, default='npo')
    parser.add_argument('--unlearn_epochs', type=int, default=15)
    parser.add_argument('--target_data_path', type=str, default='./core/data/WikiText-103-local/gpt2/random_dataset_prefixed')
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




    summarize_out_unlearn_7gram_overlap(out_data, unlearn_data, tokenizer=None, verbose=True, plot=True)
    #summarize_out_unlearn_first7gram_overlap(out_data, unlearn_data, tokenizer=None, verbose=True, plot=True)
    #compare_7gram_overlap_modes(out_data, in_data, unlearn_data, verbose=True)
    #analyze_7gram_overlap_tokenized(in_data, out_data, unlearn_data)
    #compare_first_final_7gram_overlap(out_data, in_data, verbose=True)
    compare_7gram_overlap_modes(out_data, unlearn_data, seed=42, verbose=True)

    #in_ids, out_ids, unlearn_ids = select_high_overlap_out_unl_ids_global(target_dataset, total_indices, n_grams=7, num_samples=250)
    print(f"Selected IN indices: {in_ids}")
    print(f"Selected OUT indices: {out_ids}")
    print(f"Selected UNLEARN indices: {unlearn_ids}")



    summarize_out_unlearn_7gram_overlap(out_data, unlearn_data, tokenizer=None, verbose=True, plot=True)
    analyze_7gram_overlap_tokenized(in_data, out_data, unlearn_data)
    compare_7gram_overlap_modes(out_data, unlearn_data, seed=42, verbose=True)
    run_blind_overlap_attack(out_data, unlearn_data)

    report, path = analyze_7gram_overlap_and_shift(unlearn_data, out_data, attack_dataset)
    print(f"Report: {report}")
    print(f"Path: {path}")


    results = compute_out_overlap_with_target(out_data, train_dataset)



    ########################


    train_data = ConcatDataset([in_data, unlearn_data, attack_dataset])
    results = compute_out_overlap_with_target(out_data, train_data)
    print(f"Results OUT/TRAIN: {results}")

    results = compute_out_overlap_with_target(out_data, unlearn_data)
    print(f"Results OUT/UNLEARN: {results}")



    retain_dataset = ConcatDataset([in_data, attack_dataset])

    print("[INFO] Training model on IN + UNLEARN + attack...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    train_sft(model, train_data, valid_dataset, tokenizer, args.sft_epochs)
    train_prefix(model, train_data, valid_dataset, tokenizer, args.prefix_epochs)
    original_model = copy.deepcopy(model)

    # in-exact unlearning
    # unlearned_model = unlearn_model(model, unlearn_data, retain_dataset, valid_dataset, tokenizer, args)
    # train_sft(unlearned_model, retain_dataset, valid_dataset, tokenizer, args.FT_epochs)

    # retraining
    unlearned_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    train_sft(unlearned_model, retain_dataset, valid_dataset, tokenizer, args.sft_epochs)
    train_prefix(unlearned_model, retain_dataset, valid_dataset, tokenizer, args.prefix_epochs)


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
