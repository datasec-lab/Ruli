import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import types
from tqdm import tqdm


class PrefixUnlearn(Trainer):
    def __init__(self,
                 model,
                 train_dataset,
                 retain_dataset,
                 tokenizer,
                 loss_type='ga',
                 ref_model=None,
                 beta=0.1,
                 **kwargs):

        super().__init__(model=model, train_dataset=train_dataset, **kwargs)
        self.tokenizer = tokenizer
        self.retain_dataset = retain_dataset
        self.loss_type = loss_type
        self.ref_model = ref_model.eval() if ref_model is not None else None
        self.beta = beta
        self.train_dataset = train_dataset

    @staticmethod
    def compute_last7_token_loss(logits, labels, ignore_index=-100):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Per-token cross-entropy loss, no reduction
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
            reduction='none'
        ).view(shift_labels.size())

        batch_last7_losses = []
        for i in range(loss_per_token.size(0)):
            token_losses = loss_per_token[i]
            non_masked = shift_labels[i] != ignore_index
            non_masked_indices = non_masked.nonzero(as_tuple=True)[0]
            last7_indices = non_masked_indices[-7:] if len(non_masked_indices) >= 7 else non_masked_indices
            if len(last7_indices) > 0:
                last7_loss = token_losses[last7_indices].mean().item()
                batch_last7_losses.append(last7_loss)

        if batch_last7_losses:
            avg_last7_loss = sum(batch_last7_losses) / len(batch_last7_losses)
        else:
            avg_last7_loss = 0.0

        return avg_last7_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs, retain_inputs = inputs


        if 'labels' not in forget_inputs:
            forget_inputs['labels'] = forget_inputs['input_ids'].clone()

        if 'labels' not in retain_inputs:
            retain_inputs['labels'] = retain_inputs['input_ids'].clone()

        forget_labels = forget_inputs['labels'].clone()
        for i in range(forget_labels.size(0)):
            seq_len = forget_labels[i].ne(-100).sum().item()
            if seq_len > 7:
                forget_labels[i, :-7] = -100  # mask prefix except last 7

        forget_inputs['labels'] = forget_labels

        outputs_f = model(**forget_inputs)
        loss_f = outputs_f.loss

        last7_loss = PrefixUnlearn.compute_last7_token_loss(outputs_f.logits, forget_inputs['labels'])


        if retain_inputs is not None:
            outputs_r = model(**retain_inputs)
            loss_r = outputs_r.loss
            if isinstance(loss_r, types.GeneratorType):
                loss_r = torch.stack(list(loss_r)).mean()
        else:
            loss_r = 0

        loss_f_scalar = loss_f.mean().item() if loss_f.numel() > 1 else loss_f.item()
        loss_r_scalar = loss_r.mean().item() if retain_inputs and loss_r.numel() > 1 else (
            loss_r.item() if retain_inputs else 'None')


        if self.state.global_step % 10 == 0:
            tqdm.write(
                f"[STEP {self.state.global_step}] Loss Forget: {loss_f_scalar:.4f} | Loss Retain: {loss_r_scalar:.4f}"
                f"[STEP {self.state.global_step}] Loss 7 tokens: {last7_loss:.4f}")

        if self.ref_model is not None:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(**forget_inputs)
                outputs_r_ref = self.ref_model(**retain_inputs) if retain_inputs is not None else None

        # Combine losses based on mode
        loss = 0
        if 'ga' in self.loss_type:
            loss += -loss_f
        if 'gdr' in self.loss_type:
            loss += loss_r
        if 'npo' in self.loss_type and self.ref_model:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
        if 'klr' in self.loss_type and self.ref_model:
            kl_r = F.kl_div(
                outputs_r.logits,
                outputs_r_ref.logits,
                reduction='batchmean',
                log_target=True
            )
            loss += kl_r

        if 'gadr' in self.loss_type:
            loss += -loss_f + 0.2*loss_r


        return (loss, outputs_f) if return_outputs else loss

    def get_train_dataloader(self):
        forget_loader = super().get_train_dataloader()
        retain_loader = torch.utils.data.DataLoader(
            self.retain_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
        combined = list(zip(forget_loader, retain_loader))
        self._combined_len = len(combined)  # store length for later use
        return combined

    def __len__(self):
        return self._combined_len


