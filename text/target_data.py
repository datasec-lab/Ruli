
import argparse
import random
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from datasets import Dataset
from utils import load_data

def main(args):
    # --- Load WikiText-103 validation set ---
    train_data, validation_texts, normal_texts = load_data(args.dataset_name, args)
    target_samples = random.sample(normal_texts, min(args.num_target_samples, len(normal_texts)))

    print(f"[INFO] Selected {len(target_samples)} random validation samples as target data.")

    # --- Tokenize ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_samples = []
    for text in target_samples:
        encoded = tokenizer(text, truncation=True, max_length=args.max_length)
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
    target_dataset.save_to_disk(args.save_dir)
    print(f"[INFO] Saved natural target dataset to disk at '{args.save_dir}'.")

    # Print examples of the dataset
    # for i in range(5):
    #     print(f"Example {i}:")
    #     print(tokenizer.decode(target_dataset[i]['input_ids'], skip_special_tokens=True))
    #     print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate target dataset for 7-gram unlearning.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the tokenizer model.")
    parser.add_argument("--dataset_name", type=str, default="WikiText103", help="Name of the dataset to use.")
    parser.add_argument("--num_target_samples", type=int, default=1000, help="Number of validation samples to include.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum token length for sequences.")
    parser.add_argument("--save_dir", type=str, default="./data/WikiText-103-local/pythia-70m/selective_dataset_prefixed", help="Directory to save the dataset.")
    args = parser.parse_args()

    main(args)