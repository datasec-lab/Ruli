
from utils import mul_loader, seed_everything
from utils.load_model import prepare_model
from .train import Retrain
from utils.inference import Inference
import torch
import random
import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, auc
from .train import Retrain


class MIA:
    def __init__(self, target_data, train_data, mul_test_data, args):
        self.args = args
        self.target_data = target_data
        self.train_data = train_data
        self.test_data = mul_test_data
        self.remain_data = data.Subset(self.train_data, list(range(len(self.train_data))))
        self.target_loader = data.DataLoader(self.target_data, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
        self.test_loader = data.DataLoader(self.test_data, batch_size=args.batch_size,
                                           shuffle=False, num_workers=args.num_workers)
        self.output_type = args.output_type
        self.model = prepare_model(args, fresh_seed=True)
        self.result_path = args.result_path
        self.save_dir = os.path.join(self.result_path, self.args.dataset)
        os.makedirs(self.save_dir, exist_ok=True)

    def train_shadow_model(self, model, loader_dict):
        unlearning_instance = Retrain(model, loader_dict, self.args)
        unlearning_instance.unlearn()

    def perform_inference(self, model, loader):
        inference = Inference(model, self.args)
        results = inference.get(loader)
        return results['logit_scaled_confidences']

    def train_shadow_models(self):
        N = self.args.shadow_num
        target_indices = list(range(len(self.target_data)))
        random.shuffle(target_indices)
        target_to_shadow_map = {idx: {'IN': set(), 'OUT': set()} for idx in target_indices}

        # Assign each sample to N/2 shadow models as 'IN' and N/2 shadow models as 'OUT'
        for idx in target_indices:
            shadow_indices = list(range(N))
            random.shuffle(shadow_indices)
            target_to_shadow_map[idx]['IN'] = set(shadow_indices[:N // 2])
            target_to_shadow_map[idx]['OUT'] = set(shadow_indices[N // 2:])

        in_target_logits = {idx: [] for idx in target_indices}
        out_target_logits = {idx: [] for idx in target_indices}

        for shadow_idx in range(N):
            print(f"Training shadow model {shadow_idx + 1}/{N}...")
            model = prepare_model(self.args, fresh_seed=True)
            in_samples = [idx for idx, shadows in target_to_shadow_map.items() if shadow_idx in shadows['IN']]
            out_samples = [idx for idx, shadows in target_to_shadow_map.items() if shadow_idx in shadows['OUT']]
            in_subset = data.Subset(self.target_data, in_samples)
            out_subset = data.Subset(self.target_data, out_samples)
            assert len(in_samples) > 0, f"No IN samples found for shadow model {shadow_idx + 1}"
            assert len(out_samples) > 0, f"No OUT samples found for shadow model {shadow_idx + 1}"
            combined_samples = in_samples + out_samples
            attack_data = self._generate_attack_data(combined_samples)
            if attack_data is None:
                train_data = in_subset
            else:
                train_data = data.ConcatDataset([in_subset, attack_data])
            target_loader = data.DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True,
                                            num_workers=self.args.num_workers)

            print("training data length: ", len(train_data))
            print(f"Shadow model {shadow_idx + 1}/{N} data loader length: {len(target_loader)}")
            self.train_shadow_model(model, {'train': target_loader, 'test': self.test_loader})
            print(f"Shadow model {shadow_idx + 1}/{N} trained.")
            in_logits = self.perform_inference(model, data.DataLoader(in_subset, batch_size=self.args.batch_size,
                                                                      shuffle=False, num_workers=self.args.num_workers))
            out_logits = self.perform_inference(model, data.DataLoader(out_subset,
                                                                       batch_size=self.args.batch_size, shuffle=False,
                                                                       num_workers=self.args.num_workers))

            for idx, logit in zip(in_samples, in_logits):
                in_target_logits[idx].append(logit)
            for idx, logit in zip(out_samples, out_logits):
                out_target_logits[idx].append(logit)

            print(f"Shadow model {shadow_idx + 1}/{N} inference complete.")
        return in_target_logits, out_target_logits

    def _generate_attack_data(self, exclude_samples):
        exclude_indices = set(exclude_samples)
        all_indices = set(range(len(self.train_data)))
        valid_indices = list(all_indices - exclude_indices)
        if len(valid_indices) < self.args.attack_size:
            print(
                f"Warning: "
                f"Insufficient valid samples for attack data, probably you used all the data for target samples?")
            return None

        attack_indices = random.sample(valid_indices, self.args.attack_size)
        attack_data = data.Subset(self.remain_data, attack_indices)
        return attack_data

    def collect_results(self, in_target, out_target):

        results_dict = {
            'seed': self.args.seed,
            'in': in_target,
            'out': out_target
        }
        return results_dict

    def plot_inout_distributions(self, sample_idx, in_logits, out_logits):
        # Flatten the lists of logit values
        in_logits_flat = np.concatenate(in_logits).ravel()
        out_logits_flat = np.concatenate(out_logits).ravel()
        plt.hist(in_logits_flat, bins=30, alpha=0.5, label='IN Logits', color='blue')
        plt.hist(out_logits_flat, bins=30, alpha=0.5, label='OUT Logits', color='red')
        plt.xlabel("Logit Scaled Confidence")
        plt.ylabel("Frequency")
        plt.title(f"IN vs OUT Logits for Sample {sample_idx}")
        plt.legend()


    def monitor_samples(self, in_forget_logits, out_forget_logits, sample_indices_to_monitor):
        for sample_idx in sample_indices_to_monitor:  # Ensure you iterate over each sample index
            # Ensure sample_idx is a valid key in the dictionaries
            if isinstance(sample_idx, int):
                in_logits = in_forget_logits[sample_idx]  # Access logits for the specific sample
                out_logits = out_forget_logits[sample_idx]
                self.plot_inout_distributions(sample_idx, in_logits, out_logits)
                in_mean = np.mean(in_logits)
                in_variance = np.var(in_logits)
                out_mean = np.mean(out_logits)
                out_variance = np.var(out_logits)
                print(
                    f"Sample {sample_idx} - IN Mean: {in_mean}, IN Variance: {in_variance}, OUT Mean: {out_mean}, OUT Variance: {out_variance}")
            else:
                print(f"Invalid sample index: {sample_idx}")

    def run_attack(self):
        print("\n" + "=" * 60)
        print(f"{'Starting Memory Inference Attack':^60}")
        print("=" * 60)
        print(f"{'Dataset:':<20} {self.args.dataset}")
        print(f"{'Seed:':<20} {self.args.seed}")
        print(f"{'Attack Size:':<20} {self.args.attack_size}")
        print(f"{'Shadow Models:':<20} {self.args.shadow_num}")
        print(f"{'Output Type:':<20} {self.args.output_type}")
        print(f"{'Target Data Size:':<20} {len(self.target_data)}")
        print(f"{'Model Architecture:':<20} {self.args.arch}")
        print(f"{'Device:':<20} {self.args.device}")
        print("=" * 60)
        in_target, out_target = self.train_shadow_models()
        results = self.collect_results(in_target, out_target)
        self.save_path = os.path.join(self.save_dir, f'results_{self.args.shadow_num}_{self.args.seed}_{self.args.dataset}.pth')
        torch.save(results, self.save_path)
        sample_indices_to_monitor = [self.args.example_index]  # Monitor a specific sample
        self.monitor_samples(in_target, out_target, sample_indices_to_monitor)
        return results


class MemorizedDataset(data.Dataset):
    """Custom dataset to store the most vulnerable samples."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        return self.dataset[data_idx]


class TargetModelEvaluator:
    def __init__(self, target_model, target_data, shadow_model_in_logits, shadow_model_out_logits, in_samples, args):
        self.target_model = target_model
        self.target_data = target_data
        self.args = args
        self.shadow_model_in_logits = shadow_model_in_logits
        self.shadow_model_out_logits = shadow_model_out_logits
        self.in_samples = in_samples  # List of indices that were used as IN for the target model
        self.target_loader = data.DataLoader(self.target_data, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        self.index_to_logits_map = None  # To store index mapping
        self.roc_auc = None
        self.tpr_at_0_01_fpr = None
        self.result_path = args.result_path
        self.save_dir = os.path.join(self.result_path, self.args.dataset)
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, f'results_{self.args.shadow_num}_{self.args.seed}_{self.args.dataset}.pth')

    @staticmethod
    def compute_kde_likelihood(logits, kde_estimator):
        return kde_estimator.score_samples(logits)

    def perform_inference(self):
        inference = Inference(self.target_model, self.args)
        target_logits = inference.get(self.target_loader)
        self.index_to_logits_map = {idx: logit for idx, logit in enumerate(target_logits['logit_scaled_confidences'])}
        return self.index_to_logits_map

    def evaluate_sample_likelihood(self):
        target_logits_map = self.perform_inference()
        sample_likelihoods = {}
        true_labels = []  # Store true IN/OUT labels for the ROC curve
        likelihood_ratios = []  # Store likelihood ratio scores for ROC computation

        for idx, sample_logits in target_logits_map.items():
            if idx not in self.shadow_model_in_logits or idx not in self.shadow_model_out_logits:
                print(f"Skipping sample {idx}: No corresponding shadow model results found.")
                continue

            in_sample_logits = np.array(self.shadow_model_in_logits[idx])
            out_sample_logits = np.array(self.shadow_model_out_logits[idx])

            if in_sample_logits.size == 0 or out_sample_logits.size == 0:
                print(f"Skipping sample {idx}: Empty logits found.")
                continue

            try:
                in_kde = KernelDensity(kernel='gaussian').fit(in_sample_logits.reshape(-1, 1))
                out_kde = KernelDensity(kernel='gaussian').fit(out_sample_logits.reshape(-1, 1))
                in_likelihood = np.exp(self.compute_kde_likelihood(sample_logits.reshape(-1, 1), in_kde))
                out_likelihood = np.exp(self.compute_kde_likelihood(sample_logits.reshape(-1, 1), out_kde))
                likelihood_ratio = in_likelihood / (out_likelihood + in_likelihood)
                sample_likelihoods[idx] = {
                    'in_likelihood': in_likelihood,
                    'out_likelihood': out_likelihood,
                    'likelihood_ratio': likelihood_ratio
                }
                likelihood_ratios.append(likelihood_ratio)
                true_labels.append(1 if idx in self.in_samples else 0)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")

        return sample_likelihoods, true_labels, likelihood_ratios

    @staticmethod
    def calculate_roc(true_labels, likelihood_ratios):
        fpr, tpr, thresholds = roc_curve(true_labels, likelihood_ratios)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    def plot_roc_curve(self, true_labels, likelihood_ratios):
        fpr, tpr, thresholds, roc_auc = self.calculate_roc(true_labels, likelihood_ratios)

        # Ensure roc_auc is a scalar float
        self.roc_auc = float(roc_auc)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {self.roc_auc:.2f})')
        plt.plot([1e-5, 1], [1e-5, 1], color='gray', lw=1.5, linestyle='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1e-5, 1])
        plt.ylim([1e-5, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.minorticks_on()
        dataset_name = self.args.dataset
        plt.savefig(os.path.join(self.save_dir, f'roc_curve_vulnerable_{dataset_name}.png'), dpi=700)

    def find_fpr_for_sample_count(self, true_labels, likelihood_ratios, target_sample_count):
        fpr, tpr, thresholds, roc_auc = self.calculate_roc(true_labels, likelihood_ratios)
        sorted_likelihood_ratios = sorted(likelihood_ratios, reverse=True)
        threshold_for_sample_count = float(sorted_likelihood_ratios[target_sample_count - 1])  # Convert to scalar float
        closest_idx = np.argmin(np.abs(thresholds - threshold_for_sample_count))
        fpr_at_threshold = fpr[closest_idx]
        tpr_at_threshold = tpr[closest_idx]

        print(f'Threshold for {target_sample_count} samples: {threshold_for_sample_count:.4f}')
        print(f'FPR at this threshold: {fpr_at_threshold:.4f}')
        print(f'TPR at this threshold: {tpr_at_threshold:.4f}')

        vulnerable_samples = [(i, likelihood_ratios[i], true_labels[i])
                              for i in range(len(likelihood_ratios)) if
                              likelihood_ratios[i] >= threshold_for_sample_count]

        return vulnerable_samples, fpr_at_threshold, tpr_at_threshold

    @staticmethod
    def plot_memorized_samples_roc(true_labels, likelihood_ratios, target_fpr):
        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, likelihood_ratios)

        # Find the index corresponding to the target FPR
        closest_idx = np.argmin(np.abs(fpr - target_fpr))
        threshold_at_fpr = thresholds[closest_idx]
        tpr_at_fpr = tpr[closest_idx]
        print(f"Threshold at FPR {target_fpr}: {threshold_at_fpr}")
        print(f"TPR at this FPR: {tpr_at_fpr}")
        memorized_samples = [(i, likelihood_ratios[i], true_labels[i])
                             for i in range(len(likelihood_ratios)) if likelihood_ratios[i] >= threshold_at_fpr]

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {np.trapz(tpr, fpr):.4f})', color='orange')
        plt.scatter(fpr[closest_idx], tpr_at_fpr, color='red', label=f'TPR @ FPR = {target_fpr}', zorder=5)
        for sample_idx, likelihood, label in memorized_samples:
            plt.plot(fpr[sample_idx], tpr[sample_idx], 'bo', markersize=5,
                     label=f'Memorized sample {sample_idx}' if sample_idx == 0 else "", zorder=5)

        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve with Memorized Samples @ FPR = {target_fpr}')
        plt.legend(loc='lower right')
        plt.savefig(f'scattered_roc_curve_{target_fpr}.png', dpi=700)


    def find_vulnerable_samples_at_fpr(self, true_labels, likelihood_ratios, target_fpr):
        # Calculate the ROC curve values
        fpr, tpr, thresholds, roc_auc = self.calculate_roc(true_labels, likelihood_ratios)

        # Check if target_fpr is out of range
        if target_fpr < min(fpr) or target_fpr > max(fpr):
            raise ValueError(f"target_fpr {target_fpr} is out of bounds. Min FPR: {min(fpr)}, Max FPR: {max(fpr)}")

        closest_idx = np.argmin(np.abs(fpr - target_fpr))
        if fpr[closest_idx] != target_fpr:
            if closest_idx == 0 or closest_idx == len(fpr) - 1:
                # Return closest available threshold at the boundary
                threshold_at_fpr = thresholds[closest_idx]
            else:
                # Interpolate threshold between two closest points
                fpr_left, fpr_right = fpr[closest_idx - 1], fpr[closest_idx]
                threshold_left, threshold_right = thresholds[closest_idx - 1], thresholds[closest_idx]
                threshold_at_fpr = np.interp(target_fpr, [fpr_left, fpr_right], [threshold_left, threshold_right])
                tpr_at_fpr = np.interp(target_fpr, [fpr_left, fpr_right], [tpr[closest_idx - 1], tpr[closest_idx]])
        else:
            # Exact FPR match
            threshold_at_fpr = thresholds[closest_idx]
            tpr_at_fpr = tpr[closest_idx]

        # Handle invalid threshold cases
        if np.isinf(threshold_at_fpr) or np.isnan(threshold_at_fpr):
            print(f"Invalid threshold at FPR {target_fpr}: {threshold_at_fpr}")
            return [], None, None

        # Identify vulnerable TP samples based on threshold
        vulnerable_samples = [(i, likelihood_ratios[i], true_labels[i])
                              for i in range(len(likelihood_ratios)) if
                              likelihood_ratios[i] >= threshold_at_fpr]

        vulnerable_samples = [(i, likelihood_ratios[i], true_labels[i])
                              for i in range(len(likelihood_ratios))
                              if likelihood_ratios[i] >= threshold_at_fpr and true_labels[i] == 1]


        return vulnerable_samples, threshold_at_fpr, tpr_at_fpr

    def find_privacy_preserving_samples(self, true_labels, likelihood_ratios, target_fpr, tolerance=0.00001):
        # Calculate the ROC curve values
        fpr, tpr, thresholds, roc_auc = self.calculate_roc(true_labels, likelihood_ratios)

        # Find the index of the closest FPR to the target FPR
        closest_idx = np.argmin(np.abs(fpr - target_fpr))
        threshold_at_fpr = thresholds[closest_idx]
        tpr_at_fpr = tpr[closest_idx]

        print(f"Threshold at FPR {target_fpr}: {threshold_at_fpr}")
        print(f"TPR at this FPR: {tpr_at_fpr}")
        lower_bound = 1/2 - tolerance  # e.g., 0.9
        upper_bound = 1/2 + tolerance  # e.g., 1.1
        privacy_preserving_samples = [(i, likelihood_ratios[i], true_labels[i])
                                      for i in range(len(likelihood_ratios))
                                      if lower_bound <= likelihood_ratios[i] <= upper_bound and true_labels[i] == 1]

        return privacy_preserving_samples, threshold_at_fpr, tpr_at_fpr

    def find_non_vulnerable_samples_at_fpr(self, true_labels, likelihood_ratios, target_fpr):
        fpr, tpr, thresholds, roc_auc = self.calculate_roc(true_labels, likelihood_ratios)
        closest_idx = np.argmin(np.abs(fpr - target_fpr))
        threshold_at_fpr = thresholds[closest_idx]
        tpr_at_fpr = tpr[closest_idx]
        print(f'Threshold at FPR {target_fpr}: {threshold_at_fpr:.4f}')
        print(f'TPR at this FPR: {tpr_at_fpr:.4f}')
        non_vulnerable_samples = [(i, likelihood_ratios[i], true_labels[i])
                              for i in range(len(likelihood_ratios)) if
                              likelihood_ratios[i] <= threshold_at_fpr and true_labels[i] == 1]

        return non_vulnerable_samples, threshold_at_fpr, tpr_at_fpr

    def save_vulnerable_samples(self, vulnerable_samples, dataset_filename):
        vulnerable_indices = [sample[0] for sample in vulnerable_samples]
        vulnerable_dataset = MemorizedDataset(self.target_data, vulnerable_indices)
        remaining_indices = list(set(range(len(self.target_data))) - set(vulnerable_indices))
        indices_dict = {
            'vulnerable': vulnerable_indices,
            'remaining': remaining_indices
        }

        torch.save(vulnerable_dataset, os.path.join(self.save_dir, dataset_filename + '_samples.pt'))
        torch.save(indices_dict, os.path.join(self.save_dir, dataset_filename + '_indices.pt'))
        print(f'Saved {len(vulnerable_samples)} vulnerable samples.')
        with open(os.path.join(self.save_dir, dataset_filename + '_indices.txt'), 'w') as f:
            f.write('\n'.join(map(str, vulnerable_indices)))

    def save_roc_results(self, fpr, tpr, roc):
        results_dict = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc
        }
        torch.save(results_dict,
                   os.path.join(self.save_dir, f'roc_results_{self.args.shadow_num}_{self.args.seed}_{self.args.dataset}.pth'))
        print("ROC results saved.")







