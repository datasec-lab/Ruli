import torch
import matplotlib.pyplot as plt

def plot_per_index_subplots(results_path, num_indices=120, save_path=None):
    # Load saved results
    results = torch.load(results_path)
    print(f"[INFO] Loaded result keys: {results.keys()}")

    out_unlearned = results['out_unlearned']  # {idx: [logits, ...]}
    unlearned_unlearned = results['unlearn_unlearned']  # {idx: [logits, ...]}
    in_original = results['in_original']  # {idx: [logits, ...]}
    out_original = results['out_original']
    in_unearned = results['in_unlearned']



    # {idx: [logits, ...]}

    # Select first N indices (sorted for consistency)
    common_indices = sorted(set(out_unlearned.keys()) & set(unlearned_unlearned.keys()))
    selected_indices = common_indices[:num_indices]

    print(f"[INFO] Plotting distributions for indices: {selected_indices}")

    # Setup subplots
    n_cols = 2
    n_rows = (num_indices + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        out_unl_conf = out_unlearned.get(idx, [])
        unlearned_conf = unlearned_unlearned.get(idx, [])
        in_conf = in_original.get(idx, [])
        out_conf = out_original.get(idx, [])
        in_unlearned_conf = in_unearned.get(idx, [])

        ax.hist(out_unl_conf, bins=20, alpha=0.5, density=True, label='OUT (unlearned)', color='blue')
        ax.hist(unlearned_conf, bins=20, alpha=0.5, density=True, label='UNLEARNED (unlearned)', color='orange')
        #ax.hist(in_conf, bins=20, alpha=0.5, density=True, label='IN (original)', color='green')
        ax.hist(out_conf, bins=20, alpha=0.5, density=True, label='OUT (original)', color='red')
        #ax.hist(in_unlearned_conf, bins=20, alpha=0.5, density=True, label='IN (unlearned)', color='purple')

        ax.set_title(f'Index {idx}')
        ax.set_xlabel('Logit-Scaled Confidence')
        ax.set_ylabel('Density')
        ax.legend(fontsize='small')
        ax.grid(True)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[INFO] Saved combined subplot figure to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    plot_per_index_subplots(
        '../core/attack/attack_inferences/WikiText103/shadow_60_attack_random_npo_EleutherAI_pythia-70m-deduped.pth',
        #'../core/attack/attack_inferences/WikiText103/shadow_60_attack_random_npo_gpt2.pth',
        save_path='../core/attack/attack_inferences/shadow_npo_index_random_60_pythia_random.png'
    )
#shadow_27_attack_rand
# om_npo_EleutherAI_pythia-70m-deduped.pth