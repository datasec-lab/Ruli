import os
import json
import copy
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression

from utils import mul_loader, seed_everything
from utils.load_model import prepare_model
from utils.inference import Inference
from evaluation.accuracy import eval_accuracy
from evaluation.svc_mia import basic_mia, SVC_MIA, mia_threshold
from unlearn import GradientAscent, Scrub, FineTune, GradientAscentPlus, NegGrad
from .train import Retrain

class CanaryEvaluator:
    def __init__(self, target_model, target_unlearned_model,
                 target_data, shadow_result, in_samples, unlearned_samples, out_samples,
                 args):

        self.target_model = target_model
        self.target_unlearned_model = target_unlearned_model
        self.target_data = target_data
        self.args = args
        self.in_samples = in_samples
        self.out_samples = out_samples
        self.unlearned_samples = unlearned_samples

        # Only store shadow models for indexes below self.VUL_INDEX
        self.shadow_model_in_trained = shadow_result['in_trained']
        self.shadow_model_out_trained = shadow_result['out_trained']
        self.shadow_model_unlearned_trained = shadow_result['unlearn_trained']
        self.shadow_model_out_unlearned = shadow_result['out_unlearned']
        self.shadow_model_unlearned_unlearned = shadow_result['unlearned']


        self.target_loader = data.DataLoader(self.target_data, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        self.index_to_logits_map = None  # To store index mapping for target model
        self.index_to_unl_logits_map = None  # To store index mapping for unlearned target model
        self.roc_auc = None
        self.tpr_at_0_01_fpr = None
        self.VUL_INDEX = 600  # Vulnerable index threshold

    @staticmethod
    def compute_kde_likelihood(logits, kde_estimator):
        return kde_estimator.score_samples(logits)

    def perform_inference(self):
        inference = Inference(self.target_model, self.args)
        target_logits = inference.get(self.target_loader)
        self.index_to_logits_map = {idx: logit for idx, logit in enumerate(target_logits['logit_scaled_confidences'])}
        unlearned_inference = Inference(self.target_unlearned_model, self.args)
        unlearned_target_logits = unlearned_inference.get(self.target_loader)
        self.index_to_unl_logits_map = {idx: logit for idx, logit in
                                        enumerate(unlearned_target_logits['logit_scaled_confidences'])}
        return self.index_to_logits_map, self.index_to_unl_logits_map


    @staticmethod
    def calculate_roc(true_labels, likelihood_ratios):
        fpr, tpr, _ = roc_curve(true_labels, likelihood_ratios)
        roc_auc = auc(fpr, tpr)
        fpr_at_0_01 = np.argmin(np.abs(fpr - 0.05))  # Closest FPR index to 0.01
        tpr_at_0_01_fpr = tpr[fpr_at_0_01]
        fpr_at_0_001 = np.argmin(np.abs(fpr - 0.01))  # Closest FPR index to 0.001
        tpr_at_0_001_fpr = tpr[fpr_at_0_001]
        return fpr, tpr, roc_auc, tpr_at_0_01_fpr, tpr_at_0_001_fpr


    def plot_roc_curve(self, true_labels, likelihood_ratios, plot=False):

        fpr, tpr, self.roc_auc, self.tpr_at_0_05_fpr, self.tpr_at_0_01_fpr = self.calculate_roc(true_labels, likelihood_ratios)
        attack_predictions = [1 if lr > 0.5 else 0 for lr in likelihood_ratios]
        attack_accuracy = accuracy_score(true_labels, attack_predictions)
        print(f"Attack accuracy: {attack_accuracy:.4f}")
        print(f"ROC AUC: {self.roc_auc:.4f}")
        print(f"TPR at 0.05 FPR: {self.tpr_at_0_05_fpr:.4f}")
        print(f"TPR at 0.01 FPR: {self.tpr_at_0_01_fpr:.4f}")
        print("=" * 60)
        plt.figure()
        plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {self.roc_auc:.2f})')
        plt.plot([1e-3, 1], [1e-3, 1], color='gray', lw=1.5, linestyle='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1e-3, 1])
        plt.ylim([1e-3, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.minorticks_on()
        # plt.savefig(os.path.join(self.args.result_path,
        #                          f'roc_curve_{self.args.dataset}_{self.args.task}_{self.args.unlearn_method}.png'), dpi=600)




    def evaluate_sample_likelihood(self):
        target_map, unlearned_map = self.perform_inference()
        sample_likelihoods = {}
        true_labels = []
        likelihood_ratios_org = []
        likelihood_ratios_unl = []

        for idx, sample in target_map.items():
            if idx >= self.VUL_INDEX:
                continue
            try:

                in_sample_ = np.array(self.shadow_model_in_trained[idx])
                out_sample_ = np.array(self.shadow_model_out_trained[idx])
                unl_sample_ = np.array(self.shadow_model_unlearned_trained[idx])
                unl_out_sample_ = np.array(self.shadow_model_out_unlearned[idx])
                unl_unl_sample_ = np.array(self.shadow_model_unlearned_unlearned[idx])

                in_sample_ = in_sample_[:len(unl_sample_)]
                out_sample_ = out_sample_[:len(unl_sample_)]

                in_kde = KernelDensity(kernel='gaussian').fit(in_sample_.reshape(-1, 1))
                out_kde = KernelDensity(kernel='gaussian').fit(out_sample_.reshape(-1, 1))
                unl_unl_kde = KernelDensity(kernel='gaussian').fit(unl_unl_sample_.reshape(-1, 1))
                unl_out_kde = KernelDensity(kernel='gaussian').fit(unl_out_sample_.reshape(-1, 1))

                unlearned_sample = unlearned_map[idx]
                in_likelihood = np.exp(self.compute_kde_likelihood(sample.reshape(-1, 1), in_kde))
                out_likelihood = np.exp(self.compute_kde_likelihood(sample.reshape(-1, 1), out_kde))
                lira_out_likelihood = np.exp(self.compute_kde_likelihood(unlearned_sample.reshape(-1, 1), out_kde))
                unl_likelihood = np.exp(self.compute_kde_likelihood(unlearned_sample.reshape(-1, 1), unl_unl_kde))
                unl_out_likelihood = np.exp(self.compute_kde_likelihood(unlearned_sample.reshape(-1, 1), unl_out_kde))

                in_likelihood_ratio_org = unl_likelihood / (unl_likelihood + out_likelihood)
                unl_likelihood_ratio = unl_likelihood / (unl_likelihood + unl_out_likelihood)

                sample_likelihoods[idx] = {
                    'in_likelihood': in_likelihood,
                    'out_likelihood': out_likelihood,
                    'unl_likelihood': unl_likelihood,
                    'unl_out_likelihood': unl_out_likelihood,
                    'lira_out_likelihood': lira_out_likelihood,
                }

                likelihood_ratios_unl.append(unl_likelihood_ratio)
                likelihood_ratios_org.append(in_likelihood_ratio_org)
                true_labels.append(1 if idx in self.in_samples else 0)
            except KeyError:
                continue

        return sample_likelihoods, true_labels,  likelihood_ratios_org, likelihood_ratios_unl


    def run_population_attack_vulnerable(self):
        print("Starting Population Attack for Vulnerable Samples (Indexes < self.VUL_INDEX)...")

        # Step 1: Aggregate all shadow model populations for OUT and UNLEARNED (indexes < self.VUL_INDEX)
        out_population = np.concatenate(
            [self.shadow_model_out_unlearned[idx] for idx in range(len(self.shadow_model_out_unlearned)) if idx < self.VUL_INDEX]
        ).flatten()
        unlearned_population = np.concatenate(
            [self.shadow_model_unlearned_trained[idx] for idx in range(len(self.shadow_model_unlearned_unlearned)) if
             idx < self.VUL_INDEX]
        ).flatten()

        # Create labels for populations
        out_labels = [0] * len(out_population)  # Label 0 for OUT
        unlearned_labels = [1] * len(unlearned_population)  # Label 1 for UNLEARNED

        # Combine populations and labels for training
        X_train = np.concatenate((out_population, unlearned_population)).reshape(-1, 1)
        y_train = np.array(out_labels + unlearned_labels)

        # Step 2: Train a classifier

        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)

        # Step 3: Evaluate using inferences for vulnerable samples (indexes < self.VUL_INDEX)
        _, index_to_unl_logits_map = self.perform_inference()

        # Step 3: Evaluate using inferences for the given index
        _, index_to_unl_logits_map = self.perform_inference()

        # Convert tensors to lists before concatenating
        out_samples_list = self.out_samples.tolist() if isinstance(self.out_samples, torch.Tensor) else self.out_samples
        unlearned_samples_list = self.unlearned_samples.tolist() if isinstance(self.unlearned_samples,
                                                                               torch.Tensor) else self.unlearned_samples

        # Combine the test samples
        test_population = np.array(
            [index_to_unl_logits_map[idx] for idx in (out_samples_list + unlearned_samples_list)])
        test_labels = [0 if idx in out_samples_list else 1 for idx in (out_samples_list + unlearned_samples_list)]

        y_pred = classifier.predict_proba(test_population.reshape(-1, 1))[:, 1]  # Probability of being UNLEARNED

        # Calculate metrics

        fpr, tpr, thresholds = roc_curve(test_labels, y_pred)
        auc_score = auc(fpr, tpr)
        attack_accuracy = accuracy_score(test_labels, (y_pred > 0.5).astype(int))

        # Calculate TPR@FPR=1%
        tpr_at_fpr_1 = tpr[np.argmax(fpr >= 0.01)]
        tpr_at_fpr_5 = tpr[np.argmax(fpr >= 0.05)]

        # Print results
        print("Population Attack Results for Vulnerable Samples:")
        print(f"ROC AUC: {auc_score:.4f}")
        print(f"Accuracy: {attack_accuracy:.4f}")
        print(f"TPR at FPR=1%: {tpr_at_fpr_1:.4f}")
        print(f"TPR at FPR=5%: {tpr_at_fpr_5:.4f}")
        print("=" * 60)


    def run_evaluation_LIRA(self):
        print("Starting target model evaluation...")
        sample_likelihoods, true_labels, likelihood_ratios, _ = self.evaluate_sample_likelihood()
        out_samples = torch.tensor(self.out_samples)
        unlearned_samples = torch.tensor(self.unlearned_samples)
        all_samples_indexes = torch.concat((out_samples, unlearned_samples))
        all_samples_indexes = [idx for idx in all_samples_indexes if idx < self.VUL_INDEX]

        unl_likelihood_ratios = []
        true_labels = []
        for idx in all_samples_indexes:
            idx = idx.item()
            unl_likelihood_ratios.append(sample_likelihoods[idx]['unl_likelihood'] / (sample_likelihoods[idx]['unl_likelihood'] + sample_likelihoods[idx]['lira_out_likelihood']))
            true_labels.append(1 if idx in unlearned_samples else 0)

        idx_to_likelihood = {idx.item(): ratio for idx, ratio in zip(all_samples_indexes, unl_likelihood_ratios)}
        idx_to_label = {idx.item(): label for idx, label in zip(all_samples_indexes, true_labels)}
        lower_indices = [idx.item() for idx in all_samples_indexes if idx.item() < self.VUL_INDEX]
        lower_likelihoods = [idx_to_likelihood[idx] for idx in lower_indices]
        lower_labels = [idx_to_label[idx] for idx in lower_indices]
        self.plot_roc_curve(lower_labels, lower_likelihoods)

    def run_evaluation(self):
        print("Starting target model evaluation...")
        sample_likelihoods, true_labels, _, likelihood_ratios = self.evaluate_sample_likelihood()
        out_samples = self.out_samples.detach().clone()
        unlearned_samples = self.unlearned_samples.detach().clone()


        all_samples_indexes = torch.concat((out_samples, unlearned_samples))
        all_samples_indexes = [idx for idx in all_samples_indexes if idx < self.VUL_INDEX]


        unl_likelihood_ratios = []
        true_labels = []
        for idx in all_samples_indexes:
            idx = idx.item()
            unl_likelihood_ratios.append(sample_likelihoods[idx]['unl_likelihood'] / (sample_likelihoods[idx]['unl_likelihood'] + sample_likelihoods[idx]['unl_out_likelihood']))
            true_labels.append(1 if idx in unlearned_samples else 0)

        idx_to_likelihood = {idx.item(): ratio for idx, ratio in zip(all_samples_indexes, unl_likelihood_ratios)}
        idx_to_label = {idx.item(): label for idx, label in zip(all_samples_indexes, true_labels)}
        lower_indices = [idx.item() for idx in all_samples_indexes if idx.item() < self.VUL_INDEX]
        lower_likelihoods = [idx_to_likelihood[idx] for idx in lower_indices]
        lower_labels = [idx_to_label[idx] for idx in lower_indices]
        self.plot_roc_curve(lower_labels, lower_likelihoods)
        self.run_population_attack_vulnerable()







