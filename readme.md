#  RULI: Rectifying Privacy and Efficacy Measurements in Machine Unlearning: A New Inference Attack Perspective

Welcome to **RULI**! 🎉 This repository provides a MIA attack to Rectify the Efficacy and Privacy Leakage of Unlearning


### _Note: This repository is just for reproducibility purposes and the complete code, instructions, and the latest updates will be available soon upon acceptance_


> **Quick Start**: Jump to [Usage Instructions](#-usage) to get started in minutes!

---

Before running the code, make sure you have:
-  Python 3.8 or above
-  Required dependencies (install using `conda env create -f requirements.yml -n <your_env_name>`)

---


---

## 🚀 Usage

To run the code, follow these three steps:

### 1️⃣ **Step 1 (Optional)**: Identify Protected and Safe Samples
This step is only necessary for identifying the protected and safe samples to use in the attack.

- Run Standard LIRA to get the protected and safe samples. For example, using the CIFAR10 dataset with ResNet-18 as the default:
  ```bash
  python recall_main.py --dataset cifar10 --train_epochs 50 --shadow_num 128 --device your_device --result_path your_path
  ```
- This will save the protected and safe sample indices in the specified path for use in the next steps.

---

### 2️⃣ **Step 2**: Train and Unlearn Shadow Models
Train shadow models and save the logit-scaled confidences for each sample using `unlearn_mia.py`. For example:

  ```bash
  python unlearn_mia.py --vulnerable_path your_vulnerable_file --privacy_path your_protected_file --dataset cifar10 --shadow_num 90 --device your_device --return_accuracy --task mixed --train_shadow --unlearn_method Scrub
  ```
- This step saves the shadow models and performs the attack. If you only want to perform the attack using pre-trained shadow models, skip to Step 3.

---

### 3️⃣ **Step 3**: Run the Attack
Run the attack to get the results:

  ```bash
  python unlearn_mia.py --vulnerable_path your_vulnerable_file --privacy_path your_protected_file --dataset cifar10  --device your_device --return_accuracy --task mixed --unlearn_method Scrub --saved_results your_saved_results
  ```
- This step outputs both the efficacy and privacy leakage of the unlearning method in a single run.

---

---

## 🛠 Configurations

### `--task`
Specifies the task to perform. Options include:
- `class-wise`: Random samples from a specific class.
- `selective`: Random samples from the dataset.
- `mixed`: Vulnerable and protected samples with equal sizes (**requires Step 1**).
- `canary`: Vulnerable samples and random samples with equal sizes (**requires Step 1**). 
  - *Note*: Use `canary` only for privacy leakage evaluation.

### `--unlearn_method`
Specifies the unlearning method to perform. Options include:
- `Scrub`
- `FT`: Fine-tuning/Sparse Fine-tuning
- `GA+`
- `NegGrad+`

---
Hyperparameters can be specified using the configuration file located at:
```
./core/unlearn_config.json
```

---

