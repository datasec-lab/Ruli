import torch
import torch.nn.functional as F
import copy
import math
import zlib
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.special import rel_entr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback
)

from unlearner import PrefixUnlearn

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
        output_dir='./train_prefix',
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
        output_dir="./output_sft",
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
        dataset_dir = './data/WikiText-103-local/gpt2'
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



