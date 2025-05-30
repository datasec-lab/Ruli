# import random
# from datasets import Dataset, load_dataset
# from transformers import AutoTokenizer
#TODO:canary code

# # --- Config ---
# canary_file = './canary_outputs/generated_canaries.txt'
# model_name = 'EleutherAI/pythia-70m-deduped'  # or whichever tokenizer you use
# num_prefixes = 2000  # how many WikiText-103 sentences to pull as prefixes
#
# # --- Load canaries ---
# with open(canary_file, 'r') as f:
#     canary_sequences = [line.strip() for line in f.readlines() if line.strip()]
#
# print(f"[INFO] Loaded {len(canary_sequences)} canaries from file.")
#
# # --- Load WikiText-103 validation set for prefixes ---
# raw_dataset = load_dataset('wikitext', 'wikitext-103-v1')s
# validation_texts = [item['text'].strip() for item in raw_dataset['validation'] if item['text'].strip()]
#
# # Sample a subset for prefixing (optional: you can shuffle or use all)
# prefix_pool = random.sample(validation_texts, min(num_prefixes, len(validation_texts)))
#
# print(f"[INFO] Loaded {len(prefix_pool)} WikiText-103 prefix sentences.")
#
# # --- Build prefixed canary sequences ---
# prefix_added_canaries = [
#     random.choice(prefix_pool) + " " + canary
#     for canary in canary_sequences
# ]
#
# print(f"[INFO] Created {len(prefix_added_canaries)} prefixed canary sequences.")
#
# # --- Tokenize into a Dataset ---
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#
# encoded = [tokenizer(seq, truncation=True, max_length=128) for seq in prefix_added_canaries]
#
# canary_dataset = Dataset.from_dict({
#     'input_ids': [item['input_ids'] for item in encoded],
#     'attention_mask': [item['attention_mask'] for item in encoded]
# })
#
# print(f"[INFO] Canary dataset ready: {len(canary_dataset)} samples.")
# canary_dataset.save_to_disk('canary_dataset_prefixed')
# print("[INFO] Saved prefixed canary dataset to disk at 'canary_dataset_prefixed'.")
import argparse
import random
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from utils import load_data


import random
import re
from transformers import AutoTokenizer
from datasets import Dataset
from utils import load_data

# --- Config ---
# model_name = 'EleutherAI/pythia-70m-deduped'  # or whichever tokenizer you use
# dataset_name = 'WikiText103'
# num_target_samples = 1000
#
# # --- Load data ---
# args = type('Args', (object,), {})()
# args.model_name = model_name
# train_data, tokenized_valid, normal_training_texts = load_data(dataset_name, args)
#
# # --- Filter to clean lines (remove markup) ---
# def clean_text(example):
#     text = example['text']
#     cleaned = re.sub(r'=+|#+|[*]+', '', text).strip()
#     return {'text': cleaned}
#
# tokenized_valid = tokenized_valid.map(clean_text)
#
# # --- Filter to keep only samples with ≥8 tokens and no empty text ---
# filtered_dataset = tokenized_valid.filter(
#     lambda example: len(example['input_ids']) >= 8 and example['text'].strip() != ""
# )
#
# print(f"[INFO] {len(filtered_dataset)} samples after cleaning and filtering for ≥8 tokens.")
#
# # --- Sample random indices ---
# if num_target_samples > len(filtered_dataset):
#     num_target_samples = len(filtered_dataset)
# random_indices = random.sample(range(len(filtered_dataset)), num_target_samples)
# sampled_dataset = filtered_dataset.select(random_indices)
#
# print(f"[INFO] Selected {len(sampled_dataset)} random samples for target dataset.")
#
# # --- Save to disk ---
# save_dir = './core/data/WikiText-103-local/pythia-70m/selective_dataset_prefixed'
# sampled_dataset.save_to_disk(save_dir)
# print(f"[INFO] Saved filtered target dataset to disk at '{save_dir}'.")
#
# # --- Print some decoded examples for sanity check ---
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#
# for i in range(min(10, len(sampled_dataset))):
#     decoded_text = tokenizer.decode(sampled_dataset[i]['input_ids'], skip_special_tokens=True)
#     print(f"\nExample {i + 1}:")
#     print(decoded_text)
#     print("Attention Mask:", sampled_dataset[i]['attention_mask'])
#     print("-" * 50)








#TODO:previous random sample
# --- Config ---
model_name = 'EleutherAI/pythia-160m'
num_target_samples = 1000  # number of validation samples to include
max_length = 128

args = argparse.Namespace
dataset_name = 'WikiText103'
args.model_name = model_name



# --- Load WikiText-103 validation set ---
train_data, validation_texts, normal_texts = load_data(dataset_name, args)
target_samples = random.sample(normal_texts, min(num_target_samples, len(normal_texts)))




print(f"[INFO] Selected {len(target_samples)} random validation samples as target data.")

# --- Tokenize ---
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

encoded_samples = []
for text in target_samples:
    encoded = tokenizer(text, truncation=True, max_length=max_length)
    input_ids = encoded['input_ids']

    # Only keep if long enough to have a last 7-gram
    if len(input_ids) > 7:
        encoded_samples.append({
            'input_ids': input_ids,
            'attention_mask': encoded['attention_mask']
        })

print(f"[INFO] Kept {len(encoded_samples)} samples with ≥8 tokens (for 7-gram unlearning).")

# --- Build Hugging Face Dataset ---
target_dataset = Dataset.from_dict({
    'input_ids': [item['input_ids'] for item in encoded_samples],
    'attention_mask': [item['attention_mask'] for item in encoded_samples]
})

# --- Save to disk ---
target_dataset.save_to_disk('./core/data/WikiText-103-local/pythia-70m/selective_dataset_prefixed')
print("[INFO] Saved natural target dataset to disk at 'natural_target_7gram_dataset'.")

# print examples of the dataset
for i in range(5):
    print(f"Example {i}:")
    print(tokenizer.decode(target_dataset[i]['input_ids'], skip_special_tokens=True))
    print("Attention Mask:", target_dataset[i]['attention_mask'])
    print("-" * 50)




#
#
# model_name = 'gpt2'
# num_samples = 12000
#
# # --- Load tokenizer ---
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#
# # --- Load WikiText-103 validation set ---
# raw_dataset = load_dataset('wikitext', 'wikitext-103-v1')
# validation_texts = [item['text'].strip() for item in raw_dataset['train'] if item['text'].strip()]
# # Sample a subset for prefixing (optional: you can shuffle or use all)
# validation_texts = random.sample(validation_texts, min(12000, len(validation_texts)))
# print(f"[INFO] Loaded {len(validation_texts)} validation samples total.")
#
# # --- Sample 12,000 ---
# sampled_texts = random.sample(validation_texts, min(num_samples, len(validation_texts)))
#
# total_tokens = 0
# seven_gram_counts = 0
#
# for idx, text in enumerate(sampled_texts):
#     tokens = tokenizer.encode(text, truncation=True, max_length=512)  # truncate to avoid super-long examples
#     token_count = len(tokens)
#     total_tokens += token_count
#
#     # if token_count >= 7:
#     #     seven_gram_counts += 1
#     #
#     # if (idx + 1) % 1000 == 0:
#     #     print(f"[INFO] Processed {idx + 1} samples...")
#
# print("\n=== Summary ===")
# print(f"Total samples checked: {len(sampled_texts)}")
# print(f"Total tokens across all: {total_tokens}")