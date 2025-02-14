
from utils.load_model import prepare_model
from utils.inference import Inference
import torch
import random
import copy
import torch.utils.data as data
import os
import numpy as np
from sklearn.neighbors import KernelDensity
from unlearn import  GradientAscent, Scrub, FineTune, GradientAscentPlus, NegGrad
import json
from .train import Retrain
from evaluation.accuracy import eval_accuracy
from evaluation.svc_mia import basic_mia, SVC_MIA, mia_threshold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression



class MUlMIA:
    def __init__(self, target_data, mul_test_data, attack_data, args):
        self.args = args
        self.target_data = target_data
        #self.train_data = train_data
        self.attack_data = attack_data
        self.test_data = mul_test_data
        self.target_loader = data.DataLoader(self.target_data, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
        self.test_loader = data.DataLoader(self.test_data, batch_size=args.batch_size,
                                           shuffle=False, num_workers=args.num_workers)
        self.output_type = args.output_type
        self.model = prepare_model(args, fresh_seed=True)
        self.result_path = args.result_path
        self.save_dir = os.path.join(self.result_path, self.args.dataset)
        os.makedirs(self.save_dir, exist_ok=True)
        self.unlearning_class = self.get_unlearning_method()

    def train_shadow_model(self, model, loader_dict):
        unlearning_instance = Retrain(model, loader_dict, self.args)
        unlearning_instance.unlearn()

    def unlearn_shadow_model(self, model, forget_dict, forget_data, method):
        with open('./unlearn_config.json', 'r') as f:
            all_configs = json.load(f)

        if method == 'Retrain':
            fresh_model = prepare_model(self.args, fresh_seed=True)
            loader_dict = {'train': forget_dict['remain'], 'test': forget_dict['test']}
            unlearning_instance = Retrain(fresh_model, loader_dict, self.args)
            unlearning_instance.unlearn()
            unlearned_model = copy.deepcopy(fresh_model)


        else:
            method_config = all_configs[method]
            class UnlearnArgs:
                def __init__(self, config):
                    for key, value in config.items():
                        setattr(self, key, value)

            unlearn_args = UnlearnArgs(method_config)
            unlearn_args.device = self.args.device  # Override device if necessary
            print(unlearn_args.__dict__)  # To check loaded hyperparameters
            unlearning_instance = self.unlearning_class(model, forget_dict, unlearn_args)
            unlearned_model = unlearning_instance.unlearn()

        return unlearned_model

    def perform_inference(self, model, loader):
        inference = Inference(model, self.args)
        results = inference.get(loader)
        return results['logit_scaled_confidences']


    def train_shadow_models(self):
        N = self.args.shadow_num
        target_indices = list(range(len(self.target_data)))
        print("len of target data: ", len(self.target_data))
        random.shuffle(target_indices)
        target_to_shadow_map = {idx: {'IN': set(), 'OUT': set(), 'UNLEARN': set()} for idx in target_indices}

        for idx in target_indices:
            shadow_indices = list(range(N))
            random.shuffle(shadow_indices)
            target_to_shadow_map[idx]['IN'] = set(shadow_indices[:N // 3])
            target_to_shadow_map[idx]['OUT'] = set(shadow_indices[N // 3: 2 * N // 3])
            target_to_shadow_map[idx]['UNLEARN'] = set(shadow_indices[2 * N // 3:])

        in_target_logits_original = {idx: [] for idx in target_indices}
        out_target_logits_original = {idx: [] for idx in target_indices}
        unlearn_target_logits_original = {idx: [] for idx in target_indices}
        in_target_logits_unlearned = {idx: [] for idx in target_indices}
        out_target_logits_unlearned = {idx: [] for idx in target_indices}
        unlearned_target_logits_unlearned = {idx: [] for idx in target_indices}

        for shadow_idx in range(N):
            print(f"Training shadow model {shadow_idx + 1}/{N}...")
            model = prepare_model(self.args, fresh_seed=True)
            in_samples = [idx for idx, shadows in target_to_shadow_map.items() if shadow_idx in shadows['IN']]
            out_samples = [idx for idx, shadows in target_to_shadow_map.items() if shadow_idx in shadows['OUT']]
            unlearn_samples = [idx for idx, shadows in target_to_shadow_map.items() if shadow_idx in shadows['UNLEARN']]
            assert len(in_samples) > 0, f"No IN samples found for shadow model {shadow_idx + 1}"
            assert len(out_samples) > 0, f"No OUT samples found for shadow model {shadow_idx + 1}"

            combined_samples = in_samples + unlearn_samples
            attack_data = self._generate_attack_data_mixed()
            if attack_data is None:
                train_data = data.ConcatDataset([data.Subset(self.target_data, in_samples + unlearn_samples)])
            else:
                train_data = data.ConcatDataset([data.Subset(self.target_data, in_samples + unlearn_samples),
                                                 attack_data])

            target_loader = data.DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True,
                                            num_workers=self.args.num_workers)

            self.train_shadow_model(model, {'train': target_loader, 'test': self.test_loader})
            in_logits_original = self.perform_inference(model, data.DataLoader(data.Subset(self.target_data, in_samples),
                                                                               batch_size=self.args.batch_size,
                                                                               shuffle=False, num_workers=self.args.num_workers))
            out_logits_original = self.perform_inference(model, data.DataLoader(data.Subset(self.target_data, out_samples),
                                                                                batch_size=self.args.batch_size,
                                                                                shuffle=False, num_workers=self.args.num_workers))

            unlearn_logit_original = self.perform_inference(model, data.DataLoader(data.Subset(self.target_data, unlearn_samples),
                                                                                batch_size=self.args.batch_size,
                                                                                shuffle=False, num_workers=self.args.num_workers))

            utility_data = {'forget': data.Subset(self.target_data, unlearn_samples),
                            'remain': data.Subset(self.target_data, in_samples),
                            'test': self.test_data}
            utility_loader = {'forget': data.DataLoader(data.Subset(self.target_data, unlearn_samples),
                                                         batch_size=16, shuffle=True, num_workers=self.args.num_workers),
                             'remain': data.DataLoader(data.Subset(self.target_data, in_samples),
                                                      batch_size=64, shuffle=True, num_workers=self.args.num_workers),
                             'test': self.test_loader}


            for idx, logit in zip(in_samples, in_logits_original):
                in_target_logits_original[idx].append(logit)
            for idx, logit in zip(out_samples, out_logits_original):
                out_target_logits_original[idx].append(logit)
            for idx, logit in zip(unlearn_samples, unlearn_logit_original):
                unlearn_target_logits_original[idx].append(logit)

            if len(unlearn_samples) > 0:
                forget_loader = data.DataLoader(data.Subset(self.target_data, unlearn_samples),
                                                batch_size=16, shuffle=True, num_workers=self.args.num_workers)
                print("len of forget data: ", len(data.Subset(self.target_data, unlearn_samples)))
                remain_data = data.ConcatDataset([data.Subset(self.target_data, in_samples), attack_data])
                remain_loader = data.DataLoader(remain_data, batch_size=64, shuffle=True, num_workers=self.args.num_workers)
                print("len of remain data: ", len(remain_data))

                unlearn_dict = {'forget': forget_loader, 'remain': remain_loader, 'test': self.test_loader}

                unlearn_data = {'forget': data.Subset(self.target_data, unlearn_samples),
                                'remain': remain_data,
                                'test': self.test_data}

                unlearned_model = self.unlearn_shadow_model(model, unlearn_dict, unlearn_data, self.args.unlearn_method)
                MUlMIA.unlearn_utility(unlearned_model, unlearn_dict, unlearn_data, self.args)

            in_logits_unlearned = self.perform_inference(unlearned_model, data.DataLoader(data.Subset(self.target_data, in_samples),
                                                                                batch_size=self.args.batch_size,
                                                                                shuffle=False, num_workers=self.args.num_workers))

            out_logits_unlearned = self.perform_inference(unlearned_model, data.DataLoader(data.Subset(self.target_data, out_samples),
                                                                                 batch_size=self.args.batch_size,
                                                                                 shuffle=False, num_workers=self.args.num_workers))
            unlearned_logits_unlearned = self.perform_inference(unlearned_model, data.DataLoader(data.Subset(self.target_data, unlearn_samples),
                                                                                      batch_size=self.args.batch_size,
                                                                                      shuffle=False, num_workers=self.args.num_workers))

            for idx, logit in zip(in_samples, in_logits_unlearned):
                in_target_logits_unlearned[idx].append(logit)
            for idx, logit in zip(out_samples, out_logits_unlearned):
                out_target_logits_unlearned[idx].append(logit)
            for idx, logit in zip(unlearn_samples, unlearned_logits_unlearned):
                unlearned_target_logits_unlearned[idx].append(logit)
            print(f"Shadow model {shadow_idx + 1}/{N} inference complete.")

        return (in_target_logits_original, out_target_logits_original, unlearn_target_logits_original,
                in_target_logits_unlearned,
                out_target_logits_unlearned, unlearned_target_logits_unlearned)

    def _generate_attack_data_mixed(self):
        attack_indices = random.sample(range(len(self.attack_data)), self.args.attack_size)
        attack_data = data.Subset(self.attack_data, attack_indices)
        return attack_data

    def collect_results(self, in_target_original, out_target_original, unlearn_target_original,
                        in_target_unlearned, out_target_unlearned,
                        unlearned_target_unlearned):
        results_dict = {
            'seed': self.args.seed,
            'in_trained': in_target_original,
            'out_trained': out_target_original,
            'unlearn_trained': unlearn_target_original,
            'in_unlearned': in_target_unlearned,
            'out_unlearned': out_target_unlearned,
            'unlearned': unlearned_target_unlearned
        }
        return results_dict

    def run_attack(self):
        print("\n" + "=" * 60)
        print(f"{'Starting Unleanring Inference Attack':^60}")
        print("=" * 60)
        print(f"{'Dataset:':<20} {self.args.dataset}")
        print(f"{'Seed:':<20} {self.args.seed}")
        print(f"{'Attack Size:':<20} {self.args.attack_size}")
        print(f"{'Shadow Models:':<20} {self.args.shadow_num}")
        print(f"{'Output Type:':<20} {self.args.output_type}")
        print(f"{'Target Data Size:':<20} {len(self.target_data)}")
        print(f"{'Model Architecture:':<20} {self.args.arch}")
        print(f"{'Unlearn Method:':<20} {self.args.unlearn_method}")
        print(f"{'Device:':<20} {self.args.device}")
        print("=" * 60)
        (in_target_original, out_target_original, unlearn_target_original, in_target_unlearned, out_target_unlearned,
         unlearned_target_unlearned) = self.train_shadow_models()
        results = self.collect_results(in_target_original, out_target_original, unlearn_target_original,
                                        in_target_unlearned, out_target_unlearned, unlearned_target_unlearned)
        save_path = os.path.join(self.save_dir, f'results_{self.args.shadow_num}_{self.args.seed}_{self.args.dataset}_{self.args.unlearn_method}_unlearn_{self.args.task}.pth')
        torch.save(results, save_path)
        print(f"Results saved at {save_path}")
        return results

    def get_unlearning_method(self):
        unlearning_methods = {
            'GA': GradientAscent,
            'Retrain': Retrain,
            'Scrub': Scrub,
            'FT': FineTune,
            'GA+': GradientAscentPlus,
            'NegGrad': NegGrad,
        }

        if self.args.unlearn_method not in unlearning_methods:
            raise ValueError('Invalid unlearn method')

        unlearn_method_class = unlearning_methods[self.args.unlearn_method]
        return unlearn_method_class

    @staticmethod
    def unlearn_utility(model, LOADER_DICT, DATA_DICT, args):

        if args.return_accuracy:
            print("Evaluating the model after unlearning")
            print("ACC on forget data", eval_accuracy(model, LOADER_DICT['forget'], args.device))
            print("ACC on remain data", eval_accuracy(model, LOADER_DICT['remain'], args.device))
            print("ACC on test data", eval_accuracy(model, LOADER_DICT['test'], args.device))

        # this  MIA is from https://github.com/OPTML-Group/Unlearn-Sparse.git;
        # it is not the same as the one in the paper we exclude it due to high FP rate.

        if args.return_mia_efficacy:
            print("Evaluating the MIA efficacy after unlearning")
            mia_efficacy_results = basic_mia(model, DATA_DICT['forget'], DATA_DICT['remain'], DATA_DICT['test'])
            mia_efficacy_forget = mia_efficacy_results['forget']
            mia_efficacy_remain = mia_efficacy_results['remain']
            print("MIA Efficacy Results")
            print("MIA on forget data", mia_efficacy_forget)
            print("MIA on remain data", mia_efficacy_remain)

        else:
            pass


class TargetModelEvaluator:
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
        self.shadow_model_in_trained = shadow_result['in_trained']
        self.shadow_model_out_trained = shadow_result['out_trained']
        self.shadow_model_unlearned_trained = shadow_result['unlearn_trained']
        self.shadow_model_in_unlearned = shadow_result['in_unlearned']
        self.shadow_model_out_unlearned = shadow_result['out_unlearned']
        self.shadow_model_unlearned_unlearned = shadow_result['unlearned']
        self.target_loader = data.DataLoader(self.target_data, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        self.index_to_logits_map = None  # To store index mapping for target model
        self.index_to_unl_logits_map = None  # To store index mapping for unlearned target model
        self.roc_auc = None
        self.tpr_at_0_01_fpr = None

    @staticmethod
    def compute_kde_likelihood(logits, kde_estimator):
        return kde_estimator.score_samples(logits)

    def perform_inference(self):
        inference = Inference(self.target_model, self.args)
        target_logits = inference.get(self.target_loader)
        self.index_to_logits_map = {idx: logit for idx, logit in enumerate(target_logits['logit_scaled_confidences'])}
        unlearned_inference = Inference(self.target_unlearned_model, self.args)
        unlearned_target_logits = unlearned_inference.get(self.target_loader)
        self.index_to_unl_logits_map = {idx: logit for idx, logit in enumerate(unlearned_target_logits['logit_scaled_confidences'])}
        return self.index_to_logits_map, self.index_to_unl_logits_map

    def evaluate_sample_likelihood(self):

        target_map, unlearned_map = self.perform_inference()
        sample_likelihoods = {}
        true_labels = []  # Store true IN/OUT labels for the ROC curve
        likelihood_ratios_org = []  # Store likelihood ratio scores for ROC computation
        likelihood_ratios_unl = []
        true_labels_unl = []

        for idx, sample in target_map.items():
            in_sample_ = np.array(self.shadow_model_in_trained[idx])
            out_sample_ = np.array(self.shadow_model_out_trained[idx])
            unl_sample_ = np.array(self.shadow_model_unlearned_trained[idx])
            unl_in_sample_ = np.array(self.shadow_model_in_unlearned[idx])
            unl_out_sample_ = np.array(self.shadow_model_out_unlearned[idx])
            unl_unl_sample_ = np.array(self.shadow_model_unlearned_unlearned[idx])

            in_sample_ = in_sample_[:len(unl_sample_)]
            out_sample_ = out_sample_[:len(unl_sample_)]

            in_kde = KernelDensity(kernel='gaussian').fit(in_sample_.reshape(-1, 1))
            out_kde = KernelDensity(kernel='gaussian').fit(out_sample_.reshape(-1, 1))
            unl_unl_kde = KernelDensity(kernel='gaussian').fit(unl_unl_sample_.reshape(-1, 1))
            unl_out_kde = KernelDensity(kernel='gaussian').fit(unl_out_sample_.reshape(-1, 1))
            unl_in_kde = KernelDensity(kernel='gaussian').fit(unl_in_sample_.reshape(-1, 1))
            unlearned_sample = unlearned_map[idx]
            in_likelihood = np.exp(self.compute_kde_likelihood(sample.reshape(-1, 1), in_kde))
            out_likelihood = np.exp(self.compute_kde_likelihood(sample.reshape(-1, 1), out_kde))
            unl_likelihood = np.exp(self.compute_kde_likelihood(unlearned_sample.reshape(-1, 1), unl_unl_kde))
            unl_out_likelihood = np.exp(self.compute_kde_likelihood(unlearned_sample.reshape(-1, 1), unl_out_kde))
            unl_in_likelihood = np.exp(self.compute_kde_likelihood(unlearned_sample.reshape(-1, 1), unl_in_kde))

            in_likelihood_ratio_org = in_likelihood / (in_likelihood + out_likelihood)
            unl_likelihood_ratio = unl_likelihood / (unl_likelihood + unl_out_likelihood)


            sample_likelihoods[idx] = {
                'in_likelihood': in_likelihood,
                'out_likelihood': out_likelihood,
                'unl_likelihood': unl_likelihood,
                'unl_out_likelihood': unl_out_likelihood,
                'unl_in_likelihood': unl_in_likelihood,
            }

            likelihood_ratios_unl.append(unl_likelihood_ratio)
            likelihood_ratios_org.append(in_likelihood_ratio_org)
            true_labels.append(1 if idx in self.in_samples else 0)

        return sample_likelihoods, true_labels, likelihood_ratios_org, likelihood_ratios_unl


    @staticmethod
    def calculate_roc(true_labels, likelihood_ratios):
        fpr, tpr, _ = roc_curve(true_labels, likelihood_ratios)
        roc_auc = auc(fpr, tpr)
        fpr_at_0_01 = np.argmin(np.abs(fpr - 0.01))  # Closest FPR index to 0.01
        tpr_at_0_01_fpr = tpr[fpr_at_0_01]
        fpr_at_0_001 = np.argmin(np.abs(fpr - 0.001))  # Closest FPR index to 0.001
        tpr_at_0_001_fpr = tpr[fpr_at_0_001]
        return fpr, tpr, roc_auc, tpr_at_0_01_fpr, tpr_at_0_001_fpr

    def plot_roc_curve(self, true_labels, likelihood_ratios, plot=False):

        fpr, tpr, self.roc_auc, self.tpr_at_0_01_fpr, self.tpr_at_0_001_fpr = self.calculate_roc(true_labels, likelihood_ratios)
        attack_predictions = [1 if lr > 0.5 else 0 for lr in likelihood_ratios]
        attack_accuracy = accuracy_score(true_labels, attack_predictions)
        print(f"Attack accuracy: {attack_accuracy:.4f}")
        print(f"ROC AUC: {self.roc_auc:.2f}")
        print(f"TPR at 0.01 FPR: {self.tpr_at_0_01_fpr:.2f}")
        print(f"TPR at 0.001 FPR: {self.tpr_at_0_001_fpr:.2f}")
        print("=" * 60)
        if plot:
            plt.figure()
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
            plt.savefig(os.path.join(self.args.result_path,
                                     f'roc_curve_{dataset_name}_{self.args.shadow_num}_{self.args.forget_size}'), dpi=300)


    def run_evaluation(self):
        print("═" * 50)
        print("** Privacy Leakage Attack Results **".center(50))
        print("═" * 50)
        sample_likelihoods, true_labels, _, likelihood_ratios = self.evaluate_sample_likelihood()
        #out_samples = random.sample(self.out_samples, len(self.unlearned_samples))
        out_samples = self.out_samples
        unlearned_samples = self.unlearned_samples
        in_samples = self.in_samples
        out_samples = torch.tensor(out_samples)
        unlearned_samples = torch.tensor(unlearned_samples)
        in_samples = torch.tensor(in_samples)
        all_samples_indexes = torch.concat((out_samples, unlearned_samples))
        all_samples_indexes_in = torch.concat((out_samples, in_samples))
        unl_likelihood_ratios = []
        true_labels = []

        for idx in all_samples_indexes:
            idx = idx.item()
            unl_likelihood = sample_likelihoods[idx]['unl_likelihood']
            retrain_likelihood = sample_likelihoods[idx]['unl_out_likelihood']
            unl_likelihood_ratio = unl_likelihood / (unl_likelihood + retrain_likelihood)
            unl_likelihood_ratios.append(unl_likelihood_ratio)
            true_labels.append(1 if idx in unlearned_samples else 0)

        self.plot_roc_curve(true_labels, unl_likelihood_ratios)

        if self.args.task == 'mixed' or self.args.task == 'canary':
            idx_to_likelihood = {idx.item(): ratio for idx, ratio in zip(all_samples_indexes, unl_likelihood_ratios)}
            idx_to_label = {idx.item(): label for idx, label in zip(all_samples_indexes, true_labels)}
            # we defined any index < 600 as lower group (vulnerable) and >= 600 as higher group (random/protected)
            # if using different number from utils.loader we need to change this
            lower_600_indices = [idx.item() for idx in all_samples_indexes if idx.item() < 600]
            higher_600_indices = [idx.item() for idx in all_samples_indexes if idx.item() >= 600]
            lower_600_likelihoods = [idx_to_likelihood[idx] for idx in lower_600_indices]

            #sorted_indices = sorted(lower_600_indices, key=lambda x: idx_to_likelihood[x], reverse=True)
            #print("Top 50 indices and likelihoods for lower 600")
            # for idx in sorted_indices[:50]:
            #     print(f"Index: {idx}, Likelihood: {idx_to_likelihood[idx].item():.4f}")
            lower_600_labels = [idx_to_label[idx] for idx in lower_600_indices]
            lower_600_preds = [1 if likelihood > 0.5 else 0 for likelihood in lower_600_likelihoods]
            lower_600_accuracy = sum(p == l for p, l in zip(lower_600_preds, lower_600_labels)) / len(lower_600_labels)
            higher_600_likelihoods = [idx_to_likelihood[idx] for idx in higher_600_indices]
            higher_600_labels = [idx_to_label[idx] for idx in higher_600_indices]
            higher_600_preds = [1 if likelihood > 0.5 else 0 for likelihood in higher_600_likelihoods]
            higher_600_accuracy = sum(p == l for p, l in zip(higher_600_preds, higher_600_labels)) / len(higher_600_labels)
            print(f"Accuracy for vulnerable samples: {lower_600_accuracy:.4f}")
            print(f"Accuracy for protected samples: {higher_600_accuracy:.4f}")

            lower_600_fpr, lower_600_tpr, lower_600_auc, lower_600_tpr_at_0_01, _ = self.calculate_roc(lower_600_labels, lower_600_likelihoods)
            higher_600_fpr, higher_600_tpr, higher_600_auc, higher_600_tpr_at_0_01, _ = self.calculate_roc(higher_600_labels, higher_600_likelihoods)
            print(f" vulnerable AUC: {lower_600_auc:.4f}")
            print(f" protected AUC: {higher_600_auc:.4f}")
            print(f" vulnerable TPR@FPR=0.01: {lower_600_tpr_at_0_01:.4f}")
            print(f" protected TPR@FPR=0.01: {higher_600_tpr_at_0_01:.4f}")


    def run_population_attack(self):
        print("Starting Population Attack with Global Shadow Model Populations...")

        # Step 1: Aggregate all shadow model populations for OUT and UNLEARNED
        out_population = np.concatenate(
            [self.shadow_model_out_trained[idx] for idx in range(len(self.shadow_model_out_unlearned))]
        ).flatten()
        unlearned_population = np.concatenate(
            [self.shadow_model_unlearned_trained[idx] for idx in range(len(self.shadow_model_unlearned_unlearned))]
        ).flatten()

        # Create labels for populations
        out_labels = [0] * len(out_population)  # Label 0 for OUT
        unlearned_labels = [1] * len(unlearned_population)  # Label 1 for UNLEARNED

        # Combine populations and labels for training
        X_train = np.concatenate((out_population, unlearned_population)).reshape(-1, 1)
        y_train = np.array(out_labels + unlearned_labels)

        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)

        # Step 3: Evaluate using inferences for the given index
        print("Performing inference for evaluation...")
        print("Performing inference for evaluation...")
        _, index_to_unl_logits_map = self.perform_inference()

        # Ensure self.out_samples and self.unlearned_samples are lists
        out_samples_list = self.out_samples.tolist() if isinstance(self.out_samples, torch.Tensor) else self.out_samples
        unlearned_samples_list = self.unlearned_samples.tolist() if isinstance(self.unlearned_samples,
                                                                               torch.Tensor) else self.unlearned_samples

        # Combine the lists
        test_population = np.array([index_to_unl_logits_map[idx] for idx in out_samples_list + unlearned_samples_list])
        test_labels = [0 if idx in out_samples_list else 1 for idx in out_samples_list + unlearned_samples_list]
        y_pred = classifier.predict_proba(test_population.reshape(-1, 1))[:, 1]
        fpr, tpr, thresholds = roc_curve(test_labels, y_pred)
        auc_score = auc(fpr, tpr)
        attack_accuracy = accuracy_score(test_labels, (y_pred > 0.5).astype(int))
        tpr_at_fpr_1 = tpr[np.argmax(fpr >= 0.01)]

        print("Population Attack Results:")
        print(f"ROC AUC: {auc_score:.4f}")
        print(f"Accuracy: {attack_accuracy:.4f}")
        print(f"TPR at FPR=1%: {tpr_at_fpr_1:.4f}")
        print("=" * 60)

    def run_population_attack_vulnerable(self):
        print("Starting Population Attack for Vulnerable Samples (Indexes < 600)...")

        # Step 1: Aggregate all shadow model populations for OUT and UNLEARNED (indexes < 600)
        out_population = np.concatenate(
            [self.shadow_model_out_trained[idx] for idx in range(len(self.shadow_model_out_unlearned)) if idx < 600]
        ).flatten()
        unlearned_population = np.concatenate(
            [self.shadow_model_unlearned_trained[idx] for idx in range(len(self.shadow_model_unlearned_unlearned)) if
             idx < 600]
        ).flatten()

        # Create labels for populations
        out_labels = [0] * len(out_population)  # Label 0 for OUT
        unlearned_labels = [1] * len(unlearned_population)  # Label 1 for UNLEARNED

        # Combine populations and labels for training
        X_train = np.concatenate((out_population, unlearned_population)).reshape(-1, 1)
        y_train = np.array(out_labels + unlearned_labels)

        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)

        _, index_to_unl_logits_map = self.perform_inference()
        _, index_to_unl_logits_map = self.perform_inference()

        out_samples_list = self.out_samples.tolist() if isinstance(self.out_samples, torch.Tensor) else self.out_samples
        unlearned_samples_list = self.unlearned_samples.tolist() if isinstance(self.unlearned_samples,
                                                                               torch.Tensor) else self.unlearned_samples

        test_population = np.array(
            [index_to_unl_logits_map[idx] for idx in (out_samples_list + unlearned_samples_list)])
        test_labels = [0 if idx in out_samples_list else 1 for idx in (out_samples_list + unlearned_samples_list)]

        y_pred = classifier.predict_proba(test_population.reshape(-1, 1))[:, 1]  # Probability of being UNLEARNED
        fpr, tpr, thresholds = roc_curve(test_labels, y_pred)
        auc_score = auc(fpr, tpr)
        attack_accuracy = accuracy_score(test_labels, (y_pred > 0.5).astype(int))

        # Calculate TPR@FPR=1%
        tpr_at_fpr_1 = tpr[np.argmax(fpr >= 0.01)]

        # Print results
        print("Population Attack Results for Vulnerable Samples:")
        print(f"ROC AUC: {auc_score:.4f}")
        print(f"Accuracy: {attack_accuracy:.4f}")
        print(f"TPR at FPR=1%: {tpr_at_fpr_1:.4f}")
        print("=" * 60)



class TestEvaluator:
    def __init__(self, target_model, target_unlearned_model, target_retrained_model,
                 target_data, shadow_result, in_samples, unlearned_samples, out_samples,
                 args):

        self.target_model = target_model
        self.target_unlearned_model = target_unlearned_model
        self.target_retrained_model = target_retrained_model
        self.target_data = target_data
        self.args = args
        self.in_samples = in_samples
        self.out_samples = out_samples
        self.unlearned_samples = unlearned_samples
        self.shadow_model_in_trained = shadow_result['in_trained']
        self.shadow_model_out_trained = shadow_result['out_trained']
        self.shadow_model_unlearned_trained = shadow_result['unlearn_trained']
        self.shadow_model_in_unlearned = shadow_result['in_unlearned']
        self.shadow_model_out_unlearned = shadow_result['out_unlearned']
        self.shadow_model_unlearned_unlearned = shadow_result['unlearned']
        self.target_loader = data.DataLoader(self.target_data, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
        self.index_to_logits_map = None  # To store index mapping for target model
        self.index_to_unl_logits_map = None  # To store index mapping for unlearned target model
        self.roc_auc = None
        self.tpr_at_0_01_fpr = None

    @staticmethod
    def compute_kde_likelihood(logits, kde_estimator):
        return kde_estimator.score_samples(logits)

    def calculate_roc(self, true_labels, likelihood_ratios):
        fpr, tpr, _ = roc_curve(true_labels, likelihood_ratios)
        roc_auc = auc(fpr, tpr)
        fpr_at_0_01 = np.argmin(np.abs(fpr - 0.01))  # Closest FPR index to 0.01
        tpr_at_0_01_fpr = tpr[fpr_at_0_01]
        fpr_at_0_001 = np.argmin(np.abs(fpr - 0.001))  # Closest FPR index to 0.001
        tpr_at_0_001_fpr = tpr[fpr_at_0_001]
        return fpr, tpr, roc_auc, tpr_at_0_01_fpr, tpr_at_0_001_fpr

    def plot_roc_curve_test(self, true_labels, likelihood_ratios, plot=False):
        fpr, tpr, self.roc_auc, self.tpr_at_0_01_fpr, self.tpr_at_0_001_fpr = self.calculate_roc(true_labels, likelihood_ratios)
        attack_predictions = [1 if lr > 0.5 else 0 for lr in likelihood_ratios]
        attack_accuracy = accuracy_score(true_labels, attack_predictions)
        print(f"Attack accuracy: {attack_accuracy:.4f}")
        print(f"ROC AUC: {self.roc_auc:.2f}")
        print(f"TPR at 0.01 FPR: {self.tpr_at_0_01_fpr:.2f}")
        print(f"TPR at 0.001 FPR: {self.tpr_at_0_001_fpr:.2f}")
        print("=" * 60)
        if plot:
            plt.figure()
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
            plt.savefig(os.path.join(self.args.result_path,
                                     f'roc_curve_{dataset_name}_{self.args.shadow_num}_{self.args.forget_size}'), dpi=300)


    def perform_inference(self):
        inference = Inference(self.target_model, self.args)
        target_logits = inference.get(self.target_loader)
        self.index_to_logits_map = {idx: logit for idx, logit in enumerate(target_logits['logit_scaled_confidences'])}
        unlearned_inference = Inference(self.target_unlearned_model, self.args)
        unlearned_target_logits = unlearned_inference.get(self.target_loader)
        self.index_to_unl_logits_map = {idx: logit for idx, logit in enumerate(unlearned_target_logits['logit_scaled_confidences'])}
        retrain_inference = Inference(self.target_retrained_model, self.args)
        retrain_target_logits = retrain_inference.get(self.target_loader)
        self.index_to_retrain_logits_map = {idx: logit for idx, logit in enumerate(retrain_target_logits['logit_scaled_confidences'])}
        return self.index_to_logits_map, self.index_to_unl_logits_map

    def evaluate_sample_likelihood(self):


        target_map, unlearned_map = self.perform_inference()
        sample_likelihoods = {}
        true_labels = []  # Store true IN/OUT labels for the ROC curve
        likelihood_ratios_unl = []

        for idx, sample in target_map.items():
            in_sample_ = np.array(self.shadow_model_in_trained[idx])
            out_sample_ = np.array(self.shadow_model_out_trained[idx])
            unl_sample_ = np.array(self.shadow_model_unlearned_trained[idx])
            unl_in_sample_ = np.array(self.shadow_model_in_unlearned[idx])
            unl_out_sample_ = np.array(self.shadow_model_out_unlearned[idx])
            unl_unl_sample_ = np.array(self.shadow_model_unlearned_unlearned[idx])

            in_kde = KernelDensity(kernel='gaussian').fit(in_sample_.reshape(-1, 1))
            out_kde = KernelDensity(kernel='gaussian').fit(out_sample_.reshape(-1, 1))
            unl_unl_kde = KernelDensity(kernel='gaussian').fit(unl_unl_sample_.reshape(-1, 1))
            unl_out_kde = KernelDensity(kernel='gaussian').fit(unl_out_sample_.reshape(-1, 1))
            unl_in_kde = KernelDensity(kernel='gaussian').fit(unl_in_sample_.reshape(-1, 1))


            if idx in self.unlearned_samples:
                sample = unlearned_map[idx]

            else:
                sample = target_map[idx]

            unlearn_likelihood = np.exp(self.compute_kde_likelihood(sample.reshape(-1, 1), unl_unl_kde))
            retrain_likelihood = np.exp(self.compute_kde_likelihood(sample.reshape(-1, 1), out_kde))
            out_unlearn_likelihood = np.exp(self.compute_kde_likelihood(sample.reshape(-1, 1), unl_out_kde))
            unl_likelihood_ratio = unlearn_likelihood / (unlearn_likelihood + retrain_likelihood)
            sample_likelihoods[idx] = {
                'unl_likelihood': unlearn_likelihood,
                'retrain_likelihood': retrain_likelihood,
                'unl_likelihood_ratio': unl_likelihood_ratio,
                'out_unl_likelihood': out_unlearn_likelihood
            }

            likelihood_ratios_unl.append(unl_likelihood_ratio)
            true_labels.append(1 if idx in self.in_samples else 0)


        return sample_likelihoods, true_labels, likelihood_ratios_unl


    def run_evaluation(self):
        #print("Starting target model evaluation...")
        sample_likelihoods, true_labels, likelihood_ratios = self.evaluate_sample_likelihood()
        out_samples = random.sample(self.out_samples, len(self.unlearned_samples))
        unlearned_samples = self.unlearned_samples
        out_samples = torch.tensor(out_samples)
        unlearned_samples = torch.tensor(unlearned_samples)
        all_samples_indexes = torch.concat((out_samples, unlearned_samples))
        unl_likelihood_ratios = []
        true_labels = []
        for idx in all_samples_indexes:
            idx = idx.item()
            unl_likelihood = sample_likelihoods[idx]['unl_likelihood']
            retrain_likelihood = sample_likelihoods[idx]['retrain_likelihood']
            unl_likelihood_ratio = unl_likelihood / (unl_likelihood + retrain_likelihood)
            unl_likelihood_ratios.append(unl_likelihood_ratio)
            true_labels.append(1 if idx in unlearned_samples else 0)

        print("═" * 50)
        print("** Efficacy Attack Results **".center(50))
        print("═" * 50)
        self.plot_roc_curve_test(true_labels, unl_likelihood_ratios, plot=False)
























