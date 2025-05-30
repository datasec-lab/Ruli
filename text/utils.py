import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
import math
import zlib
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainerCallback
from unlearner import PrefixUnlearn
import torch
import copy
import torch.nn.functional as F
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
import random
from sklearn.linear_model import LogisticRegression
from scipy.special import rel_entr
import matplotlib.pyplot as plt

@staticmethod
class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            perplexity = math.exp(eval_loss)
            print(f">>> Perplexity: {perplexity:.2f}")
            metrics["perplexity"] = perplexity



@staticmethod
def train_prefix(model, train_dataset, valid_dataset, tokenizer, epochs=3, batch_size=16):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ref_model = None

    training_args = TrainingArguments(
        output_dir='./unlearn_output',
        per_device_train_batch_size=4,
        num_train_epochs=epochs,
        learning_rate=1e-5,
        report_to='none',
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        #load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False
    )

    trainer = PrefixUnlearn(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        retain_dataset=train_dataset,
        loss_type='gdr',
        ref_model=ref_model,
        args=training_args,
        data_collator=lambda x: tokenizer.pad(x, return_tensors='pt'),
        eval_dataset = valid_dataset,
        callbacks=[
            #EarlyStoppingCallback(early_stopping_patience=2),
            PerplexityCallback()]
    )

    trainer.train()

    return model



@staticmethod
def train_sft(model, train_dataset, valid_dataset, tokenizer, epochs=3, batch_size=16):

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./output_canary",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir="./logs",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            PerplexityCallback()],
    )

    trainer.train()
    return model


@staticmethod
def unlearn_model(model, forget_dataset, remain_dataset, val_dataset, tokenizer, args, epochs=5, batch_size=16):
    print("[INFO] Starting unlearning (fine-tuning without UNLEARN samples)...")
    unlearn_epochs =args.unlearn_epochs
    if args.unlearn_method == 'ft':
        return train_sft(model, remain_dataset, val_dataset, tokenizer, epochs, batch_size)
    elif args.unlearn_method == 'klr' or args.unlearn_method == 'npo':
        return unlearn_prefix(model, forget_dataset, remain_dataset, tokenizer, loss_type=args.unlearn_method,
                              unlearn_epochs=unlearn_epochs,
                              ref_model=copy.deepcopy(model).to(args.device))

    else:
       return unlearn_prefix(model, forget_dataset, remain_dataset, tokenizer,
                             loss_type=args.unlearn_method, unlearn_epochs=unlearn_epochs)
#
@staticmethod
def unlearn_prefix(model, forget_dataset, retain_dataset, tokenizer, loss_type='ga', unlearn_epochs=1, ref_model=None):
    training_args = TrainingArguments(
        output_dir='./unlearn_output',
        #per_device_train_batch_size=4,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,  # 16 * 2 = 32 effective batch size
        num_train_epochs=unlearn_epochs,
        learning_rate=5e-5,
        report_to='none',
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
    )

    unlearner = PrefixUnlearn(
        model=model,
        tokenizer=tokenizer,
        train_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        loss_type=loss_type,
        ref_model= ref_model,
        args=training_args,
        data_collator=lambda x: tokenizer.pad(x, return_tensors='pt')
    )

    unlearner.train()
    #unlearner.save_model('./unlearn_output')

    return unlearner.model







@staticmethod
def inference_utils(model, tokenizer, device, monitored_canaries):

    def compute_zlib_entropy(text):
        compressed = zlib.compress(text.encode('utf-8'))
        entropy_bits = len(compressed) * 8  # total bits
        return entropy_bits / len(text)  # bits per character

    def compute_perplexity(model, tokenizer, text, device):
        inputs = tokenizer(text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
        return math.exp(loss)  # perplexity

    model.eval()
    for idx, canary in enumerate(monitored_canaries):
        inputs = tokenizer(canary, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(0)
            next_token_ids = inputs['input_ids'][0, 1:]
            target_logits = logits[:-1, :]
            true_next_logits = target_logits[torch.arange(len(next_token_ids)), next_token_ids]
            logsumexp = torch.logsumexp(target_logits, dim=-1)
            logit_scaled_conf = (true_next_logits - logsumexp).mean().item()

            print(f"\n=== Canary {idx + 1} ===")
            print(f"Text: {canary}")
            print(f"Avg LOGIT-SCALED next-token confidence: {logit_scaled_conf:.4f}")
            per_token_conf = true_next_logits - logsumexp
            for token, conf in zip(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][:-1]), per_token_conf):
                print(f"  {token} → next: {conf.item():.4f}")

        # ----- NEW: Entropy & Perplexity Metric -----
        entropy = compute_zlib_entropy(canary)
        perplexity = compute_perplexity(model, tokenizer, canary, device)
        ratio = perplexity / entropy if entropy != 0 else float('inf')

        print(f"zlib entropy (bits/char): {entropy:.4f}")
        print(f"GPT-2 perplexity: {perplexity:.4f}")
        print(f"Perplexity / Entropy ratio: {ratio:.4f}")

@staticmethod
def load_data(dataset_name, args):
    if dataset_name == 'WikiText103':
        dataset_dir = './core/data/WikiText-103-local/gpt2'
        os.makedirs(dataset_dir, exist_ok=True)
        train_cache_path = os.path.join(dataset_dir, 'tokenized_train_subset')
        valid_cache_path = os.path.join(dataset_dir, 'tokenized_valid_subset')
        #tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        if os.path.exists(train_cache_path) and os.path.exists(valid_cache_path):
            print("[INFO] Loading tokenized datasets from local disk...")
            tokenized_train = load_from_disk(train_cache_path)
            tokenized_valid = load_from_disk(valid_cache_path)
        else:
            print("[INFO] Cached subsets not found — preparing and saving them...")
            raw_dataset = load_dataset('wikitext', 'wikitext-103-v1')
            train_data = raw_dataset['train'].select(range(50000))
            valid_data = raw_dataset['validation'].select(range(2500))

            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=True, max_length=128)

            tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
            tokenized_valid = valid_data.map(tokenize_function, batched=True, remove_columns=["text"])

            tokenized_train.save_to_disk(train_cache_path)
            tokenized_valid.save_to_disk(valid_cache_path)
            print("[INFO] Tokenized datasets saved locally for future runs.")

        def remove_empty_examples(example):
            return len(example['input_ids']) > 0

        tokenized_train = tokenized_train.filter(remove_empty_examples)
        tokenized_valid = tokenized_valid.filter(remove_empty_examples)

        print("number of non empty data", len(tokenized_train))
        raw_train_data = load_dataset('wikitext', 'wikitext-103-v1')['validation'].select(range(2000))
        normal_training_texts = [item['text'] for item in raw_train_data if item['text'].strip()]

        return tokenized_train, tokenized_valid, normal_training_texts
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")



class MIAEvaluator:
    def __init__(self, target_model, unlearned_model, target_dataset, tokenizer, device, args):
        self.target_model = target_model.eval()
        self.unlearned_model = unlearned_model.eval()
        self.target_dataset = target_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.args = args



    def _batch_inference(self, model, token_lists):
        losses = []
        for input_ids_list in tqdm(token_lists, desc="Running Inference"):
            if not input_ids_list or len(input_ids_list) < 2:
                continue
            input_ids = torch.tensor(input_ids_list).unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)

            seq_len = input_ids.shape[1]
            ngram_window = min(7, seq_len - 1)
            if ngram_window <= 0:
                continue
            start_idx = max(seq_len - ngram_window - 1, 0)
            target_indices = torch.arange(start_idx, seq_len - 1)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(0)
                labels = input_ids[0, 1:]
                selected_logits = logits[:-1][target_indices]
                selected_labels = labels[target_indices]
                loss = torch.nn.functional.cross_entropy(selected_logits, selected_labels, reduction='mean').item()
            losses.append(loss)
        return losses

    def run(self, shadow_results, out_ids, unlearn_ids):
        print("[INFO] Performing inference on unlearned model")
        unlearn_token_lists = [self.target_dataset[i]['input_ids'] for i in unlearn_ids]
        out_token_lists = [self.target_dataset[i]['input_ids'] for i in out_ids]

        unlearn_losses = self._batch_inference(self.unlearned_model, unlearn_token_lists)
        out_losses = self._batch_inference(self.unlearned_model, out_token_lists)


        population_results = self.evaluate_population_attack(
            unlearn_losses=unlearn_losses,
            out_losses=out_losses,
            unlearn_ids=unlearn_ids,
            out_ids=out_ids,
            shadow_unlearn_unl=shadow_results['unlearn_unlearned'],
            shadow_out_unl=shadow_results['out_unlearned'],
        )

        print("population_results", population_results)

        return self.evaluate_with_kde(
            unlearn_losses=unlearn_losses,
            out_losses=out_losses,
            unlearn_ids=unlearn_ids,
            out_ids=out_ids,
            shadow_in=shadow_results['unlearn_unlearned'],
            shadow_out=shadow_results['out_unlearned'],
        )

    def evaluate_with_kde(self, unlearn_losses, out_losses, unlearn_ids, out_ids, shadow_in, shadow_out):
        print("[INFO] Running KDE-based likelihood ratio test (UNL vs OUT)")

        likelihood_ratios = []
        labels = []

        # Combine all samples into one list
        all_ids = unlearn_ids + out_ids
        all_losses = unlearn_losses + out_losses
        all_labels = [1] * len(unlearn_ids) + [0] * len(out_ids)

        for idx, loss, label in zip(all_ids, all_losses, all_labels):
            if idx not in shadow_in or idx not in shadow_out:
                continue

            kde_in = gaussian_kde(shadow_in[idx])
            kde_out = gaussian_kde(shadow_out[idx])

            p_in = kde_in.evaluate([loss])[0]
            p_out = kde_out.evaluate([loss])[0]
            ratio = p_in / (p_in + p_out + 1e-12)

            likelihood_ratios.append(ratio)
            labels.append(label)

            label_str = "unlearned" if label == 1 else "out"
            print(f"{label_str} {idx} p_in: {p_in:.6f}, p_out: {p_out:.6f}, ratio: {ratio:.6f}")

        # Compute evaluation metrics
        fpr, tpr, _ = roc_curve(labels, likelihood_ratios)
        # save tpr and fpr arrays into a dictionary and .pth file
        #torch.save({'tpr': tpr, 'fpr': fpr}, 'tpr_fpr_pythia_ga+gdr.pth')
        auc_score = auc(fpr, tpr)
        tpr_at_1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1] if np.any(fpr <= 0.01) else 0.0
        tpr_at_5 = tpr[np.searchsorted(fpr, 0.05, side="right") - 1] if np.any(fpr <= 0.05) else 0.0
        acc = accuracy_score(labels, np.array(likelihood_ratios) > 0.5)

        return {
            'AUC': auc_score,
            'ACC': acc,
            'TPR@1%FPR': tpr_at_1,
            'TPR@5%FPR': tpr_at_5,
            'Total': len(labels)
        }

    def evaluate_population_attack(self, unlearn_losses, out_losses, unlearn_ids, out_ids,
                                   shadow_unlearn_unl, shadow_out_unl):
        print("[INFO] Running population-level attack using shadow model outputs")

        shadow_features = []
        shadow_labels = []

        # 1. Build shadow training set
        for idx in unlearn_ids:
            if idx in shadow_unlearn_unl:
                for value in shadow_unlearn_unl[idx]:
                    shadow_features.append([value])
                    shadow_labels.append(1)

        for idx in out_ids:
            if idx in shadow_out_unl:
                for value in shadow_out_unl[idx]:
                    shadow_features.append([value])
                    shadow_labels.append(0)

        if len(shadow_features) == 0:
            print("[WARNING] No shadow data found for population attack.")
            return None

        # 2. Train a classifier on shadow model output distributions
        X_shadow = np.array(shadow_features)
        y_shadow = np.array(shadow_labels)
        clf = LogisticRegression().fit(X_shadow, y_shadow)

        # 3. Evaluate on target model losses
        X_target = np.array(unlearn_losses + out_losses).reshape(-1, 1)
        y_target = np.array([1] * len(unlearn_losses) + [0] * len(out_losses))

        y_prob = clf.predict_proba(X_target)[:, 1]

        # 4. Metrics
        fpr, tpr, _ = roc_curve(y_target, y_prob)
        auc_score = auc(fpr, tpr)
        tpr_at_1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1] if np.any(fpr <= 0.01) else 0.0
        tpr_at_5 = tpr[np.searchsorted(fpr, 0.05, side="right") - 1] if np.any(fpr <= 0.05) else 0.0
        acc = accuracy_score(y_target, y_prob > 0.5)

        return {
            'AUC': auc_score,
            'ACC': acc,
            'TPR@1%FPR': tpr_at_1,
            'TPR@5%FPR': tpr_at_5,
            'Total': len(y_target)
        }

class EfficacyEvaluator:
    def __init__(self, target_model, unlearned_model, target_dataset, tokenizer, device, args):
        self.target_model = target_model.eval()
        self.unlearned_model = unlearned_model.eval()
        self.target_dataset = target_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.args = args

    def _batch_inference(self, model, token_lists):
        losses = []
        for input_ids_list in tqdm(token_lists, desc="Running Inference"):
            if not input_ids_list or len(input_ids_list) < 2:
                continue
            input_ids = torch.tensor(input_ids_list).unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)

            seq_len = input_ids.shape[1]
            ngram_window = min(7, seq_len - 1)
            if ngram_window <= 0:
                continue
            start_idx = max(seq_len - ngram_window - 1, 0)
            target_indices = torch.arange(start_idx, seq_len - 1)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(0)
                labels = input_ids[0, 1:]
                selected_logits = logits[:-1][target_indices]
                selected_labels = labels[target_indices]
                loss = torch.nn.functional.cross_entropy(selected_logits, selected_labels, reduction='mean').item()
            losses.append(loss)
        return losses

    def run(self, shadow_results, out_ids, unlearn_ids):
        print("[INFO] Performing inference on unlearned model")
        unlearn_token_lists = [self.target_dataset[i]['input_ids'] for i in unlearn_ids]
        out_token_lists = [self.target_dataset[i]['input_ids'] for i in out_ids]

        unlearn_losses = self._batch_inference(self.unlearned_model, unlearn_token_lists)
        out_losses = self._batch_inference(self.target_model, out_token_lists)

        population_results = self.evaluate_population_attack(
            unlearn_losses=unlearn_losses,
            out_losses=out_losses,
            unlearn_ids=unlearn_ids,
            out_ids=out_ids,
            shadow_unlearn_unl=shadow_results['unlearn_unlearned'],
            shadow_out_unl=shadow_results['out_original'],
        )

        print("population_results", population_results)

        return self.evaluate_with_kde(
            unlearn_losses=unlearn_losses,
            out_losses=out_losses,
            unlearn_ids=unlearn_ids,
            out_ids=out_ids,
            shadow_in=shadow_results['unlearn_unlearned'],
            shadow_out=shadow_results['out_original'],
        )


    # def evaluate_with_kde(self, unlearn_losses, out_losses, unlearn_ids, out_ids, shadow_in, shadow_out):
    #     print("[INFO] Running KDE-based likelihood ratio test (UNL vs OUT)")
    #     likelihood_ratios = []
    #     labels = []
    #
    #     print(f"unlearn_ids: {unlearn_ids}")
    #
    #     for i, idx in enumerate(unlearn_ids):
    #         if idx not in shadow_in or idx not in shadow_out:
    #             continue
    #         kde_in = gaussian_kde(shadow_in[idx])
    #         kde_out = gaussian_kde(shadow_out[idx])
    #         p_in = kde_in.evaluate([unlearn_losses[i]])[0]
    #         p_out = kde_out.evaluate([unlearn_losses[i]])[0]
    #         ratio = p_in / (p_in + p_out + 1e-12)
    #         likelihood_ratios.append(ratio)
    #         print(f"unlearned {idx} p_in: {p_in}, p_out: {p_out}, ratio: {ratio}")
    #         labels.append(1)
    #
    #     for i, idx in enumerate(out_ids):
    #         if idx not in shadow_in or idx not in shadow_out:
    #             continue
    #         kde_in = gaussian_kde(shadow_in[idx])
    #         kde_out = gaussian_kde(shadow_out[idx])
    #         p_in = kde_in.evaluate([out_losses[i]])[0]
    #         p_out = kde_out.evaluate([out_losses[i]])[0]
    #         ratio = p_in / (p_in + p_out + 1e-12)
    #         likelihood_ratios.append(ratio)
    #         labels.append(0)
    #
    #     fpr, tpr, _ = roc_curve(labels, likelihood_ratios)
    #     auc_score = auc(fpr, tpr)
    #     tpr_at_1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1] if np.any(fpr <= 0.01) else 0.0
    #     tpr_at_5 = tpr[np.searchsorted(fpr, 0.05, side="right") - 1] if np.any(fpr <= 0.05) else 0.0
    #     acc = accuracy_score(labels, np.array(likelihood_ratios) > 0.5)
    #
    #     return {
    #         'AUC': auc_score,
    #         'ACC': acc,
    #         'TPR@1%FPR': tpr_at_1,
    #         'TPR@5%FPR': tpr_at_5,
    #         'Total': len(labels)
    #     }

    def evaluate_with_kde(self, unlearn_losses, out_losses, unlearn_ids, out_ids, shadow_in, shadow_out):
        print("[INFO] Running KDE-based likelihood ratio test (UNL vs OUT)")

        likelihood_ratios = []
        labels = []

        # Combine all samples into one list
        all_ids = unlearn_ids + out_ids
        all_losses = unlearn_losses + out_losses
        all_labels = [1] * len(unlearn_ids) + [0] * len(out_ids)

        for idx, loss, label in zip(all_ids, all_losses, all_labels):
            if idx not in shadow_in or idx not in shadow_out:
                continue

            kde_in = gaussian_kde(shadow_in[idx])
            kde_out = gaussian_kde(shadow_out[idx])

            p_in = kde_in.evaluate([loss])[0]
            p_out = kde_out.evaluate([loss])[0]
            ratio = p_in / (p_in + p_out + 1e-12)

            likelihood_ratios.append(ratio)
            labels.append(label)

            label_str = "unlearned" if label == 1 else "out"
            print(f"{label_str} {idx} p_in: {p_in:.6f}, p_out: {p_out:.6f}, ratio: {ratio:.6f}")

        # Compute evaluation metrics
        fpr, tpr, _ = roc_curve(labels, likelihood_ratios)

        "save tpr and fpr arrays into a dictionary and .pth file"
        torch.save({'tpr': tpr, 'fpr': fpr}, 'tpr_fpr_gpt2_npo.pth')


        auc_score = auc(fpr, tpr)
        tpr_at_1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1] if np.any(fpr <= 0.01) else 0.0
        tpr_at_5 = tpr[np.searchsorted(fpr, 0.05, side="right") - 1] if np.any(fpr <= 0.05) else 0.0
        acc = accuracy_score(labels, np.array(likelihood_ratios) > 0.5)

        return {
            'AUC': auc_score,
            'ACC': acc,
            'TPR@1%FPR': tpr_at_1,
            'TPR@5%FPR': tpr_at_5,
            'Total': len(labels)
        }

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc, accuracy_score
    import numpy as np

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc, accuracy_score
    import numpy as np

    def evaluate_population_attack(self, unlearn_losses, out_losses, unlearn_ids, out_ids,
                                   shadow_unlearn_unl, shadow_out_unl):
        print("[INFO] Running population-level attack using shadow model outputs")

        shadow_features = []
        shadow_labels = []

        # 1. Build shadow training set
        for idx in unlearn_ids:
            if idx in shadow_unlearn_unl:
                for value in shadow_unlearn_unl[idx]:
                    shadow_features.append([value])
                    shadow_labels.append(1)

        for idx in out_ids:
            if idx in shadow_out_unl:
                for value in shadow_out_unl[idx]:
                    shadow_features.append([value])
                    shadow_labels.append(0)

        if len(shadow_features) == 0:
            print("[WARNING] No shadow data found for population attack.")
            return None

        # 2. Train a classifier on shadow model output distributions
        X_shadow = np.array(shadow_features)
        y_shadow = np.array(shadow_labels)
        clf = LogisticRegression().fit(X_shadow, y_shadow)

        # 3. Evaluate on target model losses
        X_target = np.array(unlearn_losses + out_losses).reshape(-1, 1)
        y_target = np.array([1] * len(unlearn_losses) + [0] * len(out_losses))

        y_prob = clf.predict_proba(X_target)[:, 1]

        # 4. Metrics
        fpr, tpr, _ = roc_curve(y_target, y_prob)
        auc_score = auc(fpr, tpr)
        tpr_at_1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1] if np.any(fpr <= 0.01) else 0.0
        tpr_at_5 = tpr[np.searchsorted(fpr, 0.05, side="right") - 1] if np.any(fpr <= 0.05) else 0.0
        acc = accuracy_score(y_target, y_prob > 0.5)

        return {
            'AUC': auc_score,
            'ACC': acc,
            'TPR@1%FPR': tpr_at_1,
            'TPR@5%FPR': tpr_at_5,
            'Total': len(y_target)
        }



def summarize_out_unlearn_7gram_overlap(out_data, unlearn_data, tokenizer=None, verbose=True, plot=True):
    """
    Given tokenized OUT and UNLEARN datasets (as lists of dicts with 'input_ids'),
    compute and report:
      - Exact 7-gram overlap count and percentage
      - Mean token-level overlap across all OUT samples
      - Histogram of overlap percentages

    Args:
        out_data (List[Dict[str, List[int]]]): List of tokenized OUT samples
        unlearn_data (List[Dict[str, List[int]]]): List of tokenized UNLEARN samples
        tokenizer (Optional): If provided, can be used to decode tokens (not required)
        verbose (bool): If True, print results
        plot (bool): If True, show histogram plot

    Returns:
        Dict: {
            'exact_overlap_count': int,
            'total_out_samples': int,
            'exact_overlap_percent': float,
            'mean_token_overlap_percent': float,
            'token_overlap_list': List[float]
        }
    """
    # Extract final 7-grams
    out_7gram_list = [tuple(x['input_ids'][-7:]) for x in out_data if len(x['input_ids']) >= 7]
    unlearn_7gram_list = [tuple(x['input_ids'][-7:]) for x in unlearn_data if len(x['input_ids']) >= 7]

    out_7gram_set = set(out_7gram_list)
    unlearn_7gram_set = set(unlearn_7gram_list)

    # Exact match stats
    exact_matches = out_7gram_set & unlearn_7gram_set
    exact_overlap_count = len(exact_matches)
    total_out = len(out_7gram_set)
    exact_overlap_pct = exact_overlap_count / total_out if total_out else 0.0

    # Token-wise overlap
    tokenwise_overlap = [
        max(sum(tok in un for tok in out_gram) / 7 for un in unlearn_7gram_list)
        for out_gram in out_7gram_list
    ]
    mean_token_overlap_pct = float(np.mean(tokenwise_overlap) * 100) if tokenwise_overlap else 0.0

    if verbose:
        print(f"Exact 7-gram matches: {exact_overlap_count} / {total_out} → {exact_overlap_pct:.2%}")
        print(f"Mean token-wise 7-gram overlap: {mean_token_overlap_pct:.2f}%")

    if plot and tokenwise_overlap:
        plt.figure(figsize=(6, 4))
        plt.hist([x * 100 for x in tokenwise_overlap], bins=30, color='orchid', alpha=0.8, density=True)
        plt.xlabel('% 7-gram Token Overlap (OUT vs UNLEARN)')
        plt.ylabel('Proportion of OUT Samples')
        plt.title('Distribution of OUT–UNLEARN 7-gram Overlap')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig('out_unlearn_7gram_overlap.png')

    return {
        'exact_overlap_count': exact_overlap_count,
        'total_out_samples': total_out,
        'exact_overlap_percent': exact_overlap_pct * 100,
        'mean_token_overlap_percent': mean_token_overlap_pct,
        'token_overlap_list': tokenwise_overlap
    }


from collections import Counter

def analyze_7gram_overlap_tokenized(in_data, out_data, unlearn_data, n=7):
    """
    Computes 7-gram overlaps between IN, OUT, and UNLEARN sets.

    Each dataset is a list of dicts with pre-tokenized 'input_ids' entries.
    """
    def extract_ngrams(data):
        counter = Counter()
        for example in data:
            tokens = example['input_ids']
            if len(tokens) >= n:
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    counter[ngram] += 1
        return counter

    # Extract all 7-grams from each group
    in_ngrams = extract_ngrams(in_data)
    out_ngrams = extract_ngrams(out_data)
    unl_ngrams = extract_ngrams(unlearn_data)

    # Convert to sets
    in_set = set(in_ngrams)
    out_set = set(out_ngrams)
    unl_set = set(unl_ngrams)

    # Overlap and uniqueness
    in_out_overlap = in_set & out_set
    in_unl_overlap = in_set & unl_set
    out_unl_overlap = out_set & unl_set
    in_unique = in_set - out_set - unl_set

    # Print summary
    print(f"Total unique IN 7-grams: {len(in_set)}")
    #print(f"IN ∩ OUT overlap: {len(in_out_overlap)} ({len(in_out_overlap)/len(in_set):.2%})")
    #print(f"IN ∩ UNLEARN overlap: {len(in_unl_overlap)} ({len(in_unl_overlap)/len(in_set):.2%})")
    print(f"OUT ∩ UNLEARN overlap: {len(out_unl_overlap)} ({len(out_unl_overlap)/len(out_set):.2%})")
    #print(f"IN-only 7-grams (unique): {len(in_unique)} ({len(in_unique)/len(in_set):.2%})")

    return {
        "in_total": len(in_set),
        "in_out_overlap": len(in_out_overlap),
        "in_unl_overlap": len(in_unl_overlap),
        "in_unique": len(in_unique),
        "in_out_frac": len(in_out_overlap) / len(in_set),
        "in_unl_frac": len(in_unl_overlap) / len(in_set),
        "in_unique_frac": len(in_unique) / len(in_set),
    }


from collections import defaultdict
from tqdm import tqdm


def select_high_overlap_out_unl_ids_global(tokenized_dataset, total_indices, n_grams=7, num_samples=250):
    """
    Selects OUT and UNLEARN sample indices with highest 7-gram overlap from all indices (excluding IN).

    Args:
        tokenized_dataset: List-like dataset with 'input_ids' already tokenized.
        total_indices: Full list of sample indices.
        n_grams: n-gram size (default: 7).
        num_samples: Number of OUT and UNLEARN samples to select.

    Returns:
        in_ids (List[int])
        unlearn_ids (List[int])
        out_ids (List[int])
    """

    def extract_ngrams_for_ids(dataset, indices, n=7):
        id_to_ngrams = {}
        for idx in tqdm(indices, desc="Extracting n-grams"):
            tokens = dataset[idx]['input_ids']
            ngrams = set()
            if len(tokens) >= n:
                for i in range(len(tokens) - n + 1):
                    ngrams.add(tuple(tokens[i:i + n]))
            id_to_ngrams[idx] = ngrams
        return id_to_ngrams

    # IN samples are fixed
    in_ids = total_indices[:100]
    candidate_ids = total_indices[100:]  # All others can be OUT or UNLEARN

    # Extract 7-grams
    ngram_map = extract_ngrams_for_ids(tokenized_dataset, candidate_ids, n=n_grams)

    # Score all candidate pairs by overlap
    pair_scores = []
    for i, id1 in enumerate(tqdm(candidate_ids, desc="Scoring all pairs")):
        for id2 in candidate_ids[i + 1:]:
            overlap = len(ngram_map[id1] & ngram_map[id2])
            if overlap > 0:
                pair_scores.append((overlap, id1, id2))

    # Sort by overlap
    pair_scores.sort(reverse=True)

    # Select top unique pairs
    selected_ids = set()
    selected_pairs = []

    for _, id1, id2 in pair_scores:
        if id1 in selected_ids or id2 in selected_ids:
            continue
        selected_ids.add(id1)
        selected_ids.add(id2)
        selected_pairs.append((id1, id2))
        if len(selected_pairs) == num_samples:
            break

    # Split pairs into OUT and UNLEARN arbitrarily
    out_ids = [p[0] for p in selected_pairs]
    unlearn_ids = [p[1] for p in selected_pairs]

    return in_ids, unlearn_ids, out_ids


def compare_first_final_7gram_overlap(out_data, unlearn_data, verbose=True):
    """
    Computes and compares overlap between first and final 7-grams of OUT and UNLEARN datasets.

    Args:
        out_data: List of dicts with 'input_ids' (already tokenized)
        unlearn_data: Same format as out_data
        verbose: If True, print results

    Returns:
        Dict with first and final overlap stats
    """
    def get_first_7grams(data):
        return set(tuple(x['input_ids'][:7]) for x in data if len(x['input_ids']) >= 7)

    def get_final_7grams(data):
        return set(tuple(x['input_ids'][-7:]) for x in data if len(x['input_ids']) >= 7)

    first_out = get_first_7grams(out_data)
    first_unl = get_first_7grams(unlearn_data)
    final_out = get_final_7grams(out_data)
    final_unl = get_final_7grams(unlearn_data)

    first_overlap = len(first_out & first_unl)
    final_overlap = len(final_out & final_unl)

    stats = {
        'first_7gram_overlap_count': first_overlap,
        'first_7gram_total_out': len(first_out),
        'first_7gram_total_unl': len(first_unl),
        'first_7gram_overlap_percent': first_overlap / len(first_out) * 100 if first_out else 0.0,

        'final_7gram_overlap_count': final_overlap,
        'final_7gram_total_out': len(final_out),
        'final_7gram_total_unl': len(final_unl),
        'final_7gram_overlap_percent': final_overlap / len(final_out) * 100 if final_out else 0.0,
    }

    if verbose:
        print(f"[First 7-gram Overlap] {first_overlap} / {len(first_out)} → {stats['first_7gram_overlap_percent']:.2f}%")
        print(f"[Final 7-gram Overlap] {final_overlap} / {len(final_out)} → {stats['final_7gram_overlap_percent']:.2f}%")

    return stats


import numpy as np
import matplotlib.pyplot as plt

def summarize_out_unlearn_first7gram_overlap(out_data, unlearn_data, tokenizer=None, verbose=True, plot=True):
    """
    Given tokenized OUT and UNLEARN datasets (as lists of dicts with 'input_ids'),
    compute and report:
      - Exact first-7-gram overlap count and percentage
      - Mean token-level overlap across all OUT samples
      - Histogram of overlap percentages

    Args:
        out_data (List[Dict[str, List[int]]]): List of tokenized OUT samples
        unlearn_data (List[Dict[str, List[int]]]): List of tokenized UNLEARN samples
        tokenizer (Optional): If provided, can be used to decode tokens (not required)
        verbose (bool): If True, print results
        plot (bool): If True, show histogram plot

    Returns:
        Dict: {
            'exact_overlap_count': int,
            'total_out_samples': int,
            'exact_overlap_percent': float,
            'mean_token_overlap_percent': float,
            'token_overlap_list': List[float]
        }
    """
    # Extract first 7-grams
    out_7gram_list = [tuple(x['input_ids'][:7]) for x in out_data if len(x['input_ids']) >= 7]
    unlearn_7gram_list = [tuple(x['input_ids'][:7]) for x in unlearn_data if len(x['input_ids']) >= 7]

    out_7gram_set = set(out_7gram_list)
    unlearn_7gram_set = set(unlearn_7gram_list)

    # Exact match stats
    exact_matches = out_7gram_set & unlearn_7gram_set
    exact_overlap_count = len(exact_matches)
    total_out = len(out_7gram_set)
    exact_overlap_pct = exact_overlap_count / total_out if total_out else 0.0

    # Token-wise overlap
    tokenwise_overlap = [
        max(sum(tok in un for tok in out_gram) / 7 for un in unlearn_7gram_list)
        for out_gram in out_7gram_list
    ]
    mean_token_overlap_pct = float(np.mean(tokenwise_overlap) * 100) if tokenwise_overlap else 0.0

    if verbose:
        print(f"Exact FIRST 7-gram matches: {exact_overlap_count} / {total_out} → {exact_overlap_pct:.2%}")
        print(f"Mean token-wise FIRST 7-gram overlap: {mean_token_overlap_pct:.2f}%")

    if plot and tokenwise_overlap:
        plt.figure(figsize=(6, 4))
        plt.hist([x * 100 for x in tokenwise_overlap], bins=30, color='steelblue', alpha=0.8, density=True)
        plt.xlabel('% 7-gram Token Overlap (OUT vs UNLEARN)')
        plt.ylabel('Proportion of OUT Samples')
        plt.title('Distribution of OUT–UNLEARN First 7-gram Overlap')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig('out_unlearn_first7gram_overlap.png')

    return {
        'exact_overlap_count': exact_overlap_count,
        'total_out_samples': total_out,
        'exact_overlap_percent': exact_overlap_pct * 100,
        'mean_token_overlap_percent': mean_token_overlap_pct,
        'token_overlap_list': tokenwise_overlap
    }

import random
import numpy as np

import random
import numpy as np

def compare_7gram_overlap_modes(out_data, unlearn_data, seed=42, verbose=True):
    """
    Compare OUT vs UNLEARN 7-gram overlap for:
      - First 7 tokens
      - Final 7 tokens
      - Random 7-token span (not first/last)

    Reports both:
      - Exact match percentage
      - Mean token-wise overlap (%)

    Returns:
        Dict with exact and token-wise overlap stats for each mode
    """
    random.seed(seed)
    np.random.seed(seed)

    def extract_ngrams(data, mode):
        ngrams = []
        for sample in data:
            tokens = sample['input_ids']
            if len(tokens) < 7:
                continue
            if mode == 'first':
                ngrams.append(tuple(tokens[:7]))
            elif mode == 'final':
                ngrams.append(tuple(tokens[-7:]))
            elif mode == 'random':
                if len(tokens) >= 14:
                    start = random.randint(0, len(tokens) - 7)
                    ngrams.append(tuple(tokens[start:start+7]))
        return ngrams

    results = {}
    for mode in ['first', 'final', 'random']:
        out_grams = extract_ngrams(out_data, mode)
        unl_grams = extract_ngrams(unlearn_data, mode)

        out_set = set(out_grams)
        unl_set = set(unl_grams)

        # Exact match overlap
        exact_overlap = out_set & unl_set
        exact_pct = len(exact_overlap) / len(out_set) * 100 if out_set else 0.0

        # Token-wise overlap (best match for each OUT sample)
        tokenwise_scores = []
        for out in out_grams:
            best_score = 0
            for unl in unl_grams:
                shared = sum(tok in unl for tok in out)
                score = shared / 7
                if score > best_score:
                    best_score = score
            tokenwise_scores.append(best_score)
        mean_tokenwise = np.mean(tokenwise_scores) * 100 if tokenwise_scores else 0.0

        # Store results
        results[f'{mode}_exact_overlap_percent'] = exact_pct
        results[f'{mode}_mean_tokenwise_overlap_percent'] = mean_tokenwise

        if verbose:
            print(f"[{mode.capitalize()} 7-gram Overlap]")
            print(f"  Exact match: {len(exact_overlap)} / {len(out_set)} → {exact_pct:.2f}%")
            print(f"  Mean token-wise overlap: {mean_tokenwise:.2f}%")

    return results


import random
import numpy as np

def run_blind_overlap_attack(out_data, unlearn_data, fpr_levels=[0.01, 0.05], verbose=True, seed=42):
    """
    Run a blind attack based on best token-wise overlap of 7-grams between OUT and UNLEARN samples.

    Args:
        out_data: List of dicts with 'input_ids' (OUT set)
        unlearn_data: List of dicts with 'input_ids' (UNLEARN set)
        fpr_levels: List of FPRs to evaluate (e.g., [0.01, 0.05])
        verbose: If True, print TPR@FPR
        seed: Random seed (used if any sampling needed)

    Returns:
        Dict of TPR@FPR scores
    """
    random.seed(seed)
    np.random.seed(seed)

    # Extract 7-grams from unlearned data
    unlearn_7grams = [tuple(x['input_ids'][-7:]) for x in unlearn_data if len(x['input_ids']) >= 7]

    # Compute scores for both UNLEARN and OUT samples
    overlap_scores = []
    labels = []

    def tokenwise_score(span, comparison_grams):
        if len(span) < 7:
            return 0
        out_gram = tuple(span[-7:])
        return max(sum(tok in comp for tok in out_gram) / 7 for comp in comparison_grams) if comparison_grams else 0

    # UNLEARN = 1
    for sample in unlearn_data:
        score = tokenwise_score(sample['input_ids'], unlearn_7grams)
        overlap_scores.append(score)
        labels.append(1)

    # OUT = 0
    for sample in out_data:
        score = tokenwise_score(sample['input_ids'], unlearn_7grams)
        overlap_scores.append(score)
        labels.append(0)

    # Run threshold-based evaluation
    return blind_attack_tpr_at_fpr(overlap_scores, labels, fpr_levels=fpr_levels, verbose=verbose)



def blind_attack_tpr_at_fpr(overlap_scores, labels, fpr_levels=[0.01, 0.05], verbose=True):
    scores = np.array(overlap_scores)
    labels = np.array(labels)
    assert set(labels).issubset({0, 1})
    assert len(scores) == len(labels)

    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    total_out = np.sum(labels == 0)
    total_unl = np.sum(labels == 1)

    tpr_results = {}
    for fpr_target in sorted(fpr_levels):
        fp = 0
        tp = 0
        threshold_fp = int(np.floor(fpr_target * total_out))
        for i in range(len(scores)):
            if sorted_labels[i] == 0:
                fp += 1
            else:
                tp += 1
            if fp > threshold_fp:
                break
        tpr = tp / total_unl if total_unl else 0.0
        tpr_results[fpr_target] = tpr
        if verbose:
            print(f"Blind TPR@{int(fpr_target*100)}%FPR: {tpr:.2%}")
    return tpr_results


def analyze_7gram_distribution_shift(out_data, unlearn_data, full_dataset=None, tokenizer=None, mode='final',
                                     plot=True):
    """
    Compare token distributions in OUT and UNLEARN 7-grams to detect distribution shift.

    Args:
        out_data: list of dicts with 'input_ids'
        unlearn_data: same format
        full_dataset: optional list of all token ids to compute global frequencies
        tokenizer: optional, for printing top rare/common tokens
        mode: 'first' or 'final'
        plot: whether to show histograms

    Returns:
        dict of:
            - KL divergence
            - mean log-freq per group
            - top common/rare tokens if tokenizer is provided
    """

    def extract_7gram_tokens(data, mode):
        grams = []
        for x in data:
            tokens = x['input_ids']
            if len(tokens) < 7:
                continue
            if mode == 'first':
                grams.extend(tokens[:7])
            elif mode == 'final':
                grams.extend(tokens[-7:])
        return grams

    out_tokens = extract_7gram_tokens(out_data, mode)
    unl_tokens = extract_7gram_tokens(unlearn_data, mode)

    if full_dataset is None:
        full_token_pool = out_tokens + unl_tokens
    else:
        full_token_pool = [t for d in full_dataset for t in d['input_ids']]

    # Frequency counts
    full_counts = Counter(full_token_pool)
    out_counts = Counter(out_tokens)
    unl_counts = Counter(unl_tokens)

    vocab = list(set(out_counts) | set(unl_counts) | set(full_counts))

    # Normalize to probability distributions
    def normalize(counter, support):
        total = sum(counter.values())
        return np.array([counter.get(t, 0) / total for t in support])

    P_out = normalize(out_counts, vocab)
    P_unl = normalize(unl_counts, vocab)
    P_full = normalize(full_counts, vocab)

    # KL divergence
    kl_out_unl = np.sum(rel_entr(P_out, P_unl + 1e-10))  # KL(P_out || P_unl)

    # Mean log frequency (rarity signal)
    log_freq = np.log(np.array([full_counts.get(t, 1) for t in vocab]))
    mean_logfreq_out = np.mean([log_freq[vocab.index(t)] for t in out_tokens])
    mean_logfreq_unl = np.mean([log_freq[vocab.index(t)] for t in unl_tokens])

    if plot:
        plt.figure(figsize=(6, 4))
        plt.hist([log_freq[vocab.index(t)] for t in out_tokens], bins=30, alpha=0.6, label='OUT', color='red')
        plt.hist([log_freq[vocab.index(t)] for t in unl_tokens], bins=30, alpha=0.6, label='UNLEARN', color='blue')
        plt.xlabel('Log Token Frequency (Full Corpus)')
        plt.ylabel('Count')
        plt.title(f'Token Frequency Histogram ({mode} 7-grams)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'dist_shift_{mode}_7gram.png')

    report = {
        'KL(P_OUT || P_UNLEARN)': kl_out_unl,
        'Mean log freq (OUT)': mean_logfreq_out,
        'Mean log freq (UNLEARN)': mean_logfreq_unl,
        'OUT token count': len(out_tokens),
        'UNLEARN token count': len(unl_tokens),
    }

    if tokenizer:
        sorted_rare = sorted(vocab, key=lambda t: full_counts.get(t, 0))
        report['Top rare OUT tokens'] = [tokenizer.decode([t]) for t in sorted_rare[:5] if t in out_counts]
        report['Top rare UNLEARN tokens'] = [tokenizer.decode([t]) for t in sorted_rare[:5] if t in unl_counts]

    return report


import matplotlib.pyplot as plt
from scipy.stats import entropy
from collections import defaultdict


def analyze_7gram_overlap_and_shift(out_data, unlearn_data, full_dataset, modes=['first', 'final', 'random'], top_k=10):
    """
    Performs full analysis of 7-gram overlap and distribution shift across multiple span selection modes.
    Returns token distribution shift metrics and saves a multi-panel plot.
    """

    full_token_pool = [t for d in full_dataset for t in d['input_ids']]
    full_counts = Counter(full_token_pool)
    log_freq = {tok: np.log(freq + 1) for tok, freq in full_counts.items()}

    results = defaultdict(dict)
    num_modes = len(modes)
    fig, axs = plt.subplots(1, num_modes, figsize=(5 * num_modes, 4), sharey=True)

    for i, mode in enumerate(modes):
        out_grams = extract_7grams_by_mode(out_data, mode)
        unl_grams = extract_7grams_by_mode(unlearn_data, mode)

        out_tokens = [tok for g in out_grams for tok in g]
        unl_tokens = [tok for g in unl_grams for tok in g]

        # Token frequency distributions
        out_counts = Counter(out_tokens)
        unl_counts = Counter(unl_tokens)

        vocab = list(set(out_counts) | set(unl_counts))
        def normalize(counter):
            total = sum(counter.values())
            return np.array([counter.get(tok, 0) / total for tok in vocab])

        P_out = normalize(out_counts)
        P_unl = normalize(unl_counts)
        kl_div = np.sum(rel_entr(P_out, P_unl + 1e-10))

        # Mean log frequency
        out_logfreqs = [log_freq.get(tok, 0) for tok in out_tokens]
        unl_logfreqs = [log_freq.get(tok, 0) for tok in unl_tokens]
        mean_out = np.mean(out_logfreqs)
        mean_unl = np.mean(unl_logfreqs)

        # Exact 7-gram overlap
        out_set = set(tuple(g) for g in out_grams)
        unl_set = set(tuple(g) for g in unl_grams)
        exact_overlap = out_set & unl_set
        exact_pct = len(exact_overlap) / len(out_set) * 100 if out_set else 0

        # Plot
        axs[i].hist(out_logfreqs, bins=30, alpha=0.5, label='OUT', color='red', density=True)
        axs[i].hist(unl_logfreqs, bins=30, alpha=0.5, label='UNLEARN', color='blue', density=True)
        axs[i].set_title(f'{mode.capitalize()} 7-grams\nOverlap: {exact_pct:.1f}%, KL: {kl_div:.2f}')
        axs[i].set_xlabel('Log Token Frequency')
        axs[i].legend()

        results[mode] = {
            'exact_overlap_percent': exact_pct,
            'KL(P_OUT || P_UNLEARN)': kl_div,
            'mean_logfreq_OUT': mean_out,
            'mean_logfreq_UNLEARN': mean_unl,
            'OUT_token_count': len(out_tokens),
            'UNLEARN_token_count': len(unl_tokens)
        }

    plt.tight_layout()
    save_path = "./full_7gram_overlap_shift_analysis.png"
    plt.savefig(save_path)
    return results, save_path

def extract_7grams_by_mode(dataset, mode='final'):
    """
    Extracts a list of 7-grams from each sample in the dataset according to the given mode.

    Args:
        dataset: list of dicts with 'input_ids'
        mode: 'first', 'final', 'random', or 'all'

    Returns:
        List of 7-gram token sequences (as lists of ints)
    """
    spans = []
    for sample in dataset:
        tokens = sample['input_ids']
        if len(tokens) < 7:
            continue
        if mode == 'first':
            spans.append(tokens[:7])
        elif mode == 'final':
            spans.append(tokens[-7:])
        elif mode == 'random' and len(tokens) >= 14:
            start = random.randint(0, len(tokens) - 7)
            spans.append(tokens[start:start + 7])
        elif mode == 'all':
            for i in range(len(tokens) - 6):
                spans.append(tokens[i:i + 7])
    return spans


def compute_out_overlap_with_target(out_data, target_data, mode='final'):
    """
    Computes exact 7-gram overlap between OUT samples and the full target dataset (e.g., WikiText-103).

    Args:
        out_data: list of tokenized OUT samples (dicts with 'input_ids')
        target_data: full dataset to compare against (e.g., all target or public samples)
        mode: which 7-gram to extract ('final', 'first', or 'random')

    Returns:
        dict with:
            - number of overlaps
            - total OUT samples
            - percentage overlap
            - overlapping 7-grams (for inspection)
    """

    def get_7gram(sample, mode='final'):
        tokens = sample['input_ids']
        if len(tokens) < 7:
            return None
        if mode == 'final':
            return tuple(tokens[-7:])
        elif mode == 'first':
            return tuple(tokens[:7])
        elif mode == 'random' and len(tokens) >= 14:
            start = random.randint(0, len(tokens) - 7)
            return tuple(tokens[start:start+7])
        return None

    # Extract 7-grams from full target dataset
    target_7grams = set()
    for sample in target_data:
        gram = get_7gram(sample, mode)
        if gram:
            target_7grams.add(gram)

    # Compare OUT 7-grams to target set
    out_7grams = []
    overlap_grams = []

    for sample in out_data:
        gram = get_7gram(sample, mode)
        if gram:
            out_7grams.append(gram)
            if gram in target_7grams:
                overlap_grams.append(gram)

    overlap_count = len(overlap_grams)
    total_out = len(out_7grams)
    overlap_pct = overlap_count / total_out * 100 if total_out else 0.0

    return {
        'overlap_count': overlap_count,
        'total_out_samples': total_out,
        'overlap_percent': overlap_pct,
        'overlapping_7grams': overlap_grams[:10]  # for inspection
    }


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np

def run_bow_mia_baseline(in_data, out_data, tokenizer, max_len=32):
    """
    Trains a Bag-of-Words classifier to distinguish IN (member) vs OUT (non-member) samples.
    This is a model-free baseline to detect distribution shift in text.

    Args:
        in_data: list of dicts with 'input_ids' for IN samples (e.g., UNLEARN)
        out_data: list of dicts with 'input_ids' for OUT samples
        tokenizer: HuggingFace tokenizer (used to decode tokens)
        max_len: max number of tokens to keep per sample

    Returns:
        dict with accuracy, AUC, and detailed report
    """

    def decode_tokens(sample):
        tokens = sample['input_ids'][:max_len]
        return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    texts = [decode_tokens(x) for x in in_data] + [decode_tokens(x) for x in out_data]
    labels = [1] * len(in_data) + [0] * len(out_data)

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)

    vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_features=5000)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_bow, y_train)

    y_pred = clf.predict(X_test_bow)
    y_probs = clf.predict_proba(X_test_bow)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": acc,
        "auc": auc,
        "classification_report": report
    }



def run_bow_on_last_tokens(in_data, out_data, tokenizer, k=7):
    """
    Run Bag-of-Words MIA using only the last `k` tokens (decoded) from each sample.

    Args:
        in_data: list of IN (member) samples with 'input_ids'
        out_data: list of OUT (non-member) samples with 'input_ids'
        tokenizer: HuggingFace tokenizer to decode tokens
        k: number of final tokens to use for BoW

    Returns:
        dict: accuracy, AUC, and detailed classification report
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    def decode_last_tokens(sample):
        toks = sample['input_ids'][-k:]
        return tokenizer.decode(toks, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    texts = [decode_last_tokens(x) for x in in_data] + [decode_last_tokens(x) for x in out_data]
    labels = [1] * len(in_data) + [0] * len(out_data)

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)

    vectorizer = CountVectorizer(lowercase=True, stop_words=None, max_features=5000)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_bow, y_train)

    y_pred = clf.predict(X_test_bow)
    y_probs = clf.predict_proba(X_test_bow)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": acc,
        "auc": auc,
        "classification_report": report
    }

def compute_final_7gram_overlap_in_validation(out_data, full_validation_data):
    """
    For each OUT sample, checks whether its final 7-gram appears in the final 7-grams
    of any other sample in the validation dataset (excluding itself).

    Args:
        out_data: list of tokenized OUT samples (dicts with 'input_ids')
        full_validation_data: full validation set (as tokenized input_ids)

    Returns:
        Dictionary with:
            - overlap count (how many OUT samples have matching final 7-grams elsewhere)
            - total OUT samples
            - overlap percentage
            - overlapping 7-grams for inspection
    """
    # Build set of all final 7-grams in validation data (excluding OUT samples)
    validation_7grams = set()
    for sample in full_validation_data:
        tokens = sample['input_ids']
        if len(tokens) >= 7:
            gram = tuple(tokens[-7:])
            validation_7grams.add(gram)

    # Check overlap for each OUT sample
    out_7grams = []
    overlapping_7grams = []

    for sample in out_data:
        tokens = sample['input_ids']
        if len(tokens) < 7:
            continue
        gram = tuple(tokens[-7:])
        out_7grams.append(gram)
        if gram in validation_7grams:
            overlapping_7grams.append(gram)

    overlap_count = len(overlapping_7grams)
    total_out = len(out_7grams)
    overlap_pct = overlap_count / total_out * 100 if total_out else 0.0

    return {
        'overlap_count': overlap_count,
        'total_out_samples': total_out,
        'overlap_percent': overlap_pct,
        'overlapping_7grams': overlapping_7grams[:10]  # for inspection
    }


def compute_token_and_1gram_overlap(out_data, train_data, tokenizer):
    """
    Computes:
    - Mean token overlap percentage (token IDs)
    - Mean 1-gram word overlap percentage (decoded words)

    Args:
        out_data: list of dicts with 'input_ids' for OUT samples
        train_data: list of dicts with 'input_ids' for training data
        tokenizer: HuggingFace tokenizer (to decode tokens)

    Returns:
        Dict with:
            - mean_token_id_overlap_percent
            - mean_word_1gram_overlap_percent
            - median_word_overlap_percent
            - word_overlap_list
    """
    from collections import Counter

    # Token ID set from training data
    train_token_ids = set()
    for sample in train_data:
        train_token_ids.update(sample['input_ids'])

    # Word 1-gram set from training data
    train_words = set()
    for sample in train_data:
        text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        train_words.update(text.split())

    # Calculate overlaps
    token_id_overlap_percents = []
    word_overlap_percents = []

    for sample in out_data:
        ids = sample['input_ids']
        if not ids:
            continue

        # Token ID overlap
        token_overlap_count = sum(1 for t in ids if t in train_token_ids)
        token_id_overlap_percents.append(token_overlap_count / len(ids) * 100)

        # 1-gram word overlap
        text = tokenizer.decode(ids, skip_special_tokens=True)
        words = text.split()
        if not words:
            continue
        word_overlap_count = sum(1 for w in words if w in train_words)
        word_overlap_percents.append(word_overlap_count / len(words) * 100)

    return {
        'mean_token_id_overlap_percent': float(np.mean(token_id_overlap_percents)),
        'mean_word_1gram_overlap_percent': float(np.mean(word_overlap_percents)),
        'median_word_1gram_overlap_percent': float(np.median(word_overlap_percents)),
        'word_overlap_list': word_overlap_percents
    }