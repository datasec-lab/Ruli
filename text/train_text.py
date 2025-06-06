from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch
import random
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.nn.functional as F
from utils import unlearn_prefix

class LanguageMIA:
    def __init__(self, target_dataset, valid_dataset, attack_dataset, tokenizer, args):
        self.target_dataset = target_dataset
        self.valid_dataset = valid_dataset
        self.attack_dataset = attack_dataset
        self.args = args
        self.tokenizer = tokenizer

    def train_shadow_models(self):
        N = self.args.shadow_num
        target_indices = list(range(len(self.target_dataset)))

        # Assign IN / OUT / UNLEARN per target
        target_to_shadow_map = {
            idx: {'IN': set(), 'OUT': set(), 'UNLEARN': set()}
            for idx in target_indices
        }
        for idx in target_indices:
            shadow_indices = list(range(N))
            random.shuffle(shadow_indices)
            target_to_shadow_map[idx]['IN'] = set(shadow_indices[:N // 3])
            target_to_shadow_map[idx]['OUT'] = set(shadow_indices[N // 3: 2 * N // 3])
            target_to_shadow_map[idx]['UNLEARN'] = set(shadow_indices[2 * N // 3:])

        # Store logits/confidences
        in_logits_original = {idx: [] for idx in target_indices}
        out_logits_original = {idx: [] for idx in target_indices}
        unlearn_logits_original = {idx: [] for idx in target_indices}
        in_logits_unlearned = {idx: [] for idx in target_indices}
        out_logits_unlearned = {idx: [] for idx in target_indices}
        unlearn_logits_unlearned = {idx: [] for idx in target_indices}

        for shadow_idx in range(N):
            print(f"\n[INFO] Shadow Model {shadow_idx + 1}/{N}")
            torch.cuda.empty_cache()
            gc.collect()

            # Assign samples for this shadow
            in_ids = [idx for idx, shadows in target_to_shadow_map.items() if shadow_idx in shadows['IN']]
            out_ids = [idx for idx, shadows in target_to_shadow_map.items() if shadow_idx in shadows['OUT']]
            unlearn_ids = [idx for idx, shadows in target_to_shadow_map.items() if shadow_idx in shadows['UNLEARN']]

            combined_ids = in_ids + unlearn_ids
            combined_train = ConcatDataset([Subset(self.target_dataset, combined_ids)])
            if self.attack_dataset:
                combined_train = ConcatDataset([combined_train, self.attack_dataset])


            model = self.prepare_fresh_model()
            self.args.sft_epochs = 5
            self.train_sft(model, combined_train, self.valid_dataset, self.tokenizer, self.args)
            self.train_prefix(model, combined_train, self.valid_dataset, self.tokenizer, self.args)

            in_token_lists = [self.target_dataset[idx]['input_ids'] for idx in in_ids]
            out_token_lists = [self.target_dataset[idx]['input_ids'] for idx in out_ids]
            unlearn_token_lists = [self.target_dataset[idx]['input_ids'] for idx in unlearn_ids]

            in_logits = self.perform_lm_inference(model, in_token_lists, self.tokenizer, self.args.device)
            out_logits = self.perform_lm_inference(model, out_token_lists, self.tokenizer, self.args.device)
            unlearn_logits = self.perform_lm_inference(model, unlearn_token_lists, self.tokenizer, self.args.device)

            for idx, logit in zip(in_ids, in_logits):
                in_logits_original[idx].append(logit)
            for idx, logit in zip(out_ids, out_logits):
                out_logits_original[idx].append(logit)
            for idx, logit in zip(unlearn_ids, unlearn_logits):
                unlearn_logits_original[idx].append(logit)

            # Unlearn (fine-tune without unlearn_ids)
            remain_ids = in_ids
            remain_train = ConcatDataset([Subset(self.target_dataset, remain_ids)])
            forget_train = ConcatDataset([Subset(self.target_dataset, unlearn_ids)])
            if self.attack_dataset:
                remain_train = ConcatDataset([remain_train, self.attack_dataset])

            unlearned_model = self.unlearn_model(model, forget_train, remain_train, self.valid_dataset, self.tokenizer,
                                                 self.args)


            self.args.sft_epochs = 2
            self.train_sft(unlearned_model, remain_train, self.valid_dataset, self.tokenizer, self.args)

            # Inference after unlearning
            in_logits_unl = self.perform_lm_inference(unlearned_model, in_token_lists, self.args.tokenizer, self.args.device)
            out_logits_unl = self.perform_lm_inference(unlearned_model, out_token_lists, self.args.tokenizer, self.args.device)
            unlearned_logits_unl = self.perform_lm_inference(unlearned_model, unlearn_token_lists, self.args.tokenizer,
                                                             self.args.device)

            for idx, logit in zip(in_ids, in_logits_unl):
                in_logits_unlearned[idx].append(logit)
            for idx, logit in zip(out_ids, out_logits_unl):
                out_logits_unlearned[idx].append(logit)
            for idx, logit in zip(unlearn_ids, unlearned_logits_unl):
                unlearn_logits_unlearned[idx].append(logit)

        return {
            'in_original': in_logits_original,
            'out_original': out_logits_original,
            'unlearn_original': unlearn_logits_original,
            'in_unlearned': in_logits_unlearned,
            'out_unlearned': out_logits_unlearned,
            'unlearn_unlearned': unlearn_logits_unlearned
        }

    def prepare_fresh_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.args.model_name).to(self.args.device)
        return model

    def train_sft(self, model, train_dataset, valid_dataset, tokenizer, args):
        from utils import train_sft  # Your utils function
        epochs = args.sft_epochs
        return train_sft(model, train_dataset, valid_dataset,tokenizer, epochs)

    def train_prefix(self, model, train_dataset, valid_dataset, tokenizer, args):
        from utils import train_prefix
        epochs = args.prefix_epochs
        return train_prefix(model, train_dataset, valid_dataset, tokenizer, epochs)


    def unlearn_model(self, model, forget_dataset, remain_dataset, valid_dataset, tokenizer, args):
        from utils import unlearn_model  # Your utils function
        return unlearn_model(model, forget_dataset, remain_dataset, valid_dataset, tokenizer, args)


    def perform_lm_inference(self, model, token_id_batches, tokenizer, device):
        losses = []

        for idx, input_ids_list in enumerate(token_id_batches):
            if not input_ids_list or len(input_ids_list) < 2:
                print(f"[WARNING] Skipping empty or too-short input at idx {idx}")
                continue

            input_ids = torch.tensor(input_ids_list).unsqueeze(0).to(device)  # [1, seq_len]
            attention_mask = torch.ones_like(input_ids).to(device)

            seq_len = input_ids.shape[1]
            ngram_window = min(7, seq_len - 1)
            if ngram_window <= 0:
                continue

            start_idx = seq_len - ngram_window - 1
            if start_idx < 0:
                start_idx = 0

            target_indices = torch.arange(start_idx, seq_len - 1)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(0)  # [seq_len, vocab]
                labels = input_ids[0, 1:]  # next-token labels [seq_len - 1]

                selected_logits = logits[:-1][target_indices]  # [ngram, vocab]
                selected_labels = labels[target_indices]  # [ngram]

                loss = F.cross_entropy(
                    selected_logits,
                    selected_labels,
                    reduction='mean'
                ).item()

            losses.append(loss)

        return losses
