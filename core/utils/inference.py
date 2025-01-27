import torch
import numpy as np
import torch.nn.functional as F
import argparse
import torch
import numpy as np
import torch.nn as nn


class EncoderInference:
    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.encoder = self.get_encoder()
        self.args = args

    def get_encoder(self):
        """
        Extracts the encoder (feature extractor) from the ResNet model by removing the final classification layer.
        """
        return nn.Sequential(*list(self.model.children())[:-1])  # Remove the classifier layer

    def get(self, loader):
        embeddings = []  # Per-example embeddings
        labels = []  # Ground truth labels

        self.encoder.eval()

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Get encoder features
                features = self.encoder(inputs)
                features = features.view(features.size(0), -1)  # Flatten features

                # Store per-example embeddings and labels
                embeddings.extend(features.cpu().numpy())
                #labels.extend(targets.cpu().numpy())

        # Return results as a dictionary
        results = {
            'embeddings': np.array(embeddings),  # Per-example encoder embeddings
            #'labels': np.array(labels)  # Ground truth labels
        }
        return results



class LogitInference:

    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.args = args
        self.output_type = args.output_type

    def get(self, loader):
        self.model.eval()
        self.criterion = DistillKL(1)
        logits = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.model(inputs)
                if output.dim() == 1:
                    output = output.unsqueeze(1)

                logits.extend(output.cpu().numpy())  # Avoid flattening unnecessarily

        results = {
            'logits': np.array(logits)
        }

        return results



class DistillInference:

    def __init__(self, unl_model, org_model, args):
        self.model = unl_model
        self.org_model = org_model
        self.device = args.device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.args = args
        self.output_type = args.output_type
        self.Inference = Inference(self.model, self.args)

    def get(self, loader):
        logits = []
        losses = []
        logits_org = []
        confidences = []  # Confidence of the true class
        logit_scaled_confidences = []
        logits_kl = []
        entropies = []
        self.model.eval()
        self.org_model.eval()
        self.criterion = DistillKL(1)

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.model(inputs)
                output_org = self.org_model(inputs)

                # Ensure outputs have at least 2 dimensions
                if output.dim() == 1:
                    output = output.unsqueeze(1)
                if output_org.dim() == 1:
                    output_org = output_org.unsqueeze(1)

                logits.extend(output.cpu().numpy())  # Avoid flattening unnecessarily
                logits_org.extend(output_org.cpu().numpy())  # Keep logits structure

                # Calculate per-sample loss
                self.criterion.reduction = 'none'
                loss_per_sample = self.criterion(output, output_org)
                losses.extend(loss_per_sample.cpu().numpy().tolist())  # Append per-sample losses


        results =  self.Inference.get(loader)
        results['Distill'] = np.array(losses)
        results['logits_org'] = np.array(logits_org)
        # # Prepare results dictionary
        # results = {
        #     'logits': np.array(logits),
        #     'losses': np.array(losses),
        #     'confidences': np.array(confidences),  # Confidence of the true class
        #     'logit_scaled_confidences': np.array(logit_scaled_confidences),
        #     'entropies': np.array(entropies)
        # }

        return results


class DistillKL(nn.Module):

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        # Compute per-sample KL divergence
        loss_per_sample = F.kl_div(p_s, p_t, reduction='none').sum(dim=1)
        loss = loss_per_sample * (self.T ** 2)
        return loss


class Inference:

    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.args = args
        self.output_type = args.output_type

    def get(self, loader):
        logits = []
        losses = []
        confidences = []  # Confidence of the true class
        logit_scaled_confidences = []
        logit_scaled_confidences_adapted = []
        entropies = []
        labels = []
        self.model.eval()

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.model(inputs)

                # Collect logits
                logits.extend(output.cpu().numpy().flatten())
                loss = self.criterion(output, targets).item()
                losses.extend([loss] * len(targets))  # Repeat the loss for each sample in the batch

                probs = torch.nn.functional.softmax(output, dim=1)
                true_class_confidences = probs[range(probs.shape[0]), targets].cpu().numpy()
                confidences.extend(true_class_confidences)  # Store the true class confidence

                eps = 1e-6
                log_f_x_y = np.log(np.clip(true_class_confidences, eps, 1 - eps))  # Log of true class confidence
                other_probs = probs.clone()
                other_probs[range(other_probs.shape[0]), targets] = 0  # Zero out the probability of the true class
                log_other_probs_sum = np.log(
                    np.clip(other_probs.sum(dim=1).cpu().numpy(), eps, None))  # Sum of all other probabilities
                stable_logit_confidence = log_f_x_y - log_other_probs_sum
                logit_scaled_confidences.extend(stable_logit_confidence)


                _, predicted = torch.max(output.data, 1)
                predicted_confidences = probs[range(probs.shape[0]), predicted].cpu().numpy()
                log_f_x_pred = np.log(np.clip(predicted_confidences, eps, 1 - eps))  # Log of predicted confidence
                other_probs_pred = probs.clone()
                other_probs_pred[range(other_probs_pred.shape[0]), predicted] = 0  # Zero out predicted class probability
                log_other_probs_sum_pred = np.log(
                    np.clip(other_probs_pred.sum(dim=1).cpu().numpy(), eps, None))  # Sum of all other probabilities
                stable_logit_confidence_pred = log_f_x_pred - log_other_probs_sum_pred
                logit_scaled_confidences_adapted.extend(stable_logit_confidence_pred)

                # Compute entropy
                log_probs = torch.nn.functional.log_softmax(output, dim=1)
                entropy = -torch.sum(probs * log_probs, dim=1).cpu().numpy()  # Standard entropy formula
                entropies.extend(entropy)

                # Collect labels
                labels.extend(predicted.cpu().numpy())

                # Compute entropy
                log_probs = torch.nn.functional.log_softmax(output, dim=1)
                entropy = -torch.sum(probs * log_probs, dim=1).cpu().numpy()  # Standard entropy formula
                entropies.extend(entropy)

                # compute the labels
                _, predicted = torch.max(output.data, 1)
                labels.extend(predicted.cpu().numpy())

        # Prepare results dictionary
        results = {
            'logits': np.array(logits),
            'losses': np.array(losses),
            'confidences': np.array(confidences),  # Confidence of the true class
            'logit_scaled_confidences': np.array(logit_scaled_confidences),
            'entropies': np.array(entropies),
            'labels': np.array(labels),
            'logit_scaled_confidences_adapted': np.array(logit_scaled_confidences),

        }

        return results
















































    # def get(self, loader):
    #     logits = []
    #     losses = []
    #     confidences = []
    #     logit_scaled_confidences = []
    #     entropies = []
    #     self.model.eval()
    #     with torch.no_grad():
    #         for inputs, targets in loader:
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             output = self.model(inputs)
    #             logits.extend(output.cpu().numpy().flatten())
    #             loss = self.criterion(output, targets).item()
    #             losses.extend([loss] * len(targets))  # Repeat the loss for each sample in the batch
    #             probs = torch.nn.functional.softmax(output, dim=1)
    #             confidence, _ = probs.max(dim=1)
    #             confidences.extend(confidence.cpu().numpy().flatten())
    #
    #             # Stable logit scaled confidence calculation
    #             eps = 1e-6
    #             f_x_y = probs[range(probs.shape[0]), targets].cpu().numpy()
    #             log_f_x_y = np.log(np.clip(f_x_y, eps, 1 - eps))
    #             other_probs = probs.clone()
    #             other_probs[range(other_probs.shape[0]), targets] = 0  # Set the correct label probability to 0
    #             log_other_probs_sum = np.log(np.sum(other_probs.cpu().numpy(), axis=1))
    #             stable_logit_confidence = log_f_x_y - log_other_probs_sum
    #             logit_scaled_confidences.extend(stable_logit_confidence)
    #
    #             log_probs = torch.nn.functional.log_softmax(output, dim=1)
    #             probs = torch.exp(log_probs)
    #             entropy = -torch.sum(probs * log_probs, dim=1).cpu().numpy()
    #             entropies.extend(entropy)
    #
    #     results = {
    #         'logits': np.array(logits),
    #         'losses': np.array(losses),
    #         'confidences': np.array(confidences),
    #         'logit_scaled_confidences': np.array(logit_scaled_confidences),
    #         'entropies': np.array(entropies)
    #     }
    #
    #     return results