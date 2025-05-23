import numpy as np
import matplotlib.pyplot as plt

# Constants
NUM_BINS = 2**16
BIN_EDGES = np.linspace(-1, 1, NUM_BINS)
FontSize_title = 36
FontSize_legend = 24
axis_fontsize = 36


import numpy as np
import matplotlib.pyplot as plt

def plot_precomputed_pmfs(pmf_a, pmf_b, label_a, label_b, title, bins=2**16, save_path=None):
    """
    Plot two precomputed PMFs side by side with non-blocking behavior for scripts.

    Parameters:
    - pmf_a: np.ndarray — PMF values A
    - pmf_b: np.ndarray — PMF values B
    - label_a: str — legend label for A
    - label_b: str — legend label for B
    - title: str — plot title
    - bins: int — number of bins (default: 65536)
    - save_path: str — if given, saves plot to file
    - pause_time: float — how long to display the plot (seconds)
    """
    assert len(pmf_a) == bins and len(pmf_b) == bins, "PMF arrays must match bin count."

    bin_edges = np.linspace(-1, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(14, 6))
    plt.bar(bin_centers, pmf_a, alpha=0.5, width=1, label=label_a)
    plt.bar(bin_centers, pmf_b, alpha=0.5, width=1, label=label_b)
    plt.xlabel('bins')
    plt.ylabel('PMF')
    plt.xlim([-0.01, 0.01])
    plt.ylim([0, 0.01])
    plt.title(title, fontsize=24)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
  






if __name__ == "__main__":
   
    # === ASVspoof ASVspoof5 === 
    pmf_probs_bonafide_train_ASVspoof05 = np.load('./ASVspoof5_train/train_data_ASVSpoof5_pmf_probs_bonafide.npy')
    pmf_probs_spoofed_train_ASVspoof05  = np.load('./ASVspoof5_train/train_data_ASVSpoof5_pmf_probs_spoofed.npy')

    # pmf_probs_bonafide_dev_ASVspoof05 = np.load('./ASVspoof5_dev/dev_data_ASVSpoof5_pmf_probs_bonafide.npy')
    # pmf_probs_spoofed_dev_ASVspoof05  = np.load('./ASVspoof5_dev/dev_data_ASVSpoof5_pmf_probs_spoofed.npy')

    # pmf_probs_bonafide_eval_ASVspoof05 = np.load('./ASVspoof5_eval/eval_data_ASVSpoof5_pmf_probs_bonafide.npy')
    # pmf_probs_spoofed_eval_ASVspoof05  = np.load('./ASVspoof5_eval/eval_data_ASVSpoof5_pmf_probs_spoofed.npy')

    # # === ASVspoof 2019 ===
    # pmf_probs_bonafide_train_ASVspoof2019 = np.load('./ASVspoof2019_train/train_data_ASVSpoof2019_pmf_probs_bonafide.npy')
    # pmf_probs_spoofed_train_ASVspoof2019  = np.load('./ASVspoof2019_train/train_data_ASVSpoof2019_pmf_probs_spoofed.npy')

    # pmf_probs_bonafide_dev_ASVspoof2019 = np.load('./ASVspoof2019_dev/dev_data_ASVSpoof2019_pmf_probs_bonafide.npy')
    # pmf_probs_spoofed_dev_ASVspoof2019  = np.load('./ASVspoof2019_dev/dev_data_ASVSpoof2019_pmf_probs_spoofed.npy')

    # pmf_probs_bonafide_eval_ASVspoof2019 = np.load('./ASVspoof2019_eval/eval_data_ASVSpoof2019_pmf_probs_bonafide.npy')
    # pmf_probs_spoofed_eval_ASVspoof2019  = np.load('./ASVspoof2019_eval/eval_data_ASVSpoof2019_pmf_probs_spoofed.npy')

     # === ASVspoof05 ===

    # Train
    plot_precomputed_pmfs(
        pmf_probs_bonafide_train_ASVspoof05,
        pmf_probs_spoofed_train_ASVspoof05,
        label_a='ASVspoof05 Train Bonafide',
        label_b='ASVspoof05 Train Spoofed',
        title='PMF of bonafide vs spoof in the train ASVspoof05 database'
    )

    # Dev
    plot_precomputed_pmfs(
        pmf_probs_bonafide_dev_ASVspoof05,
        pmf_probs_spoofed_dev_ASVspoof05,
        label_a='ASVspoof05 Dev Bonafide',
        label_b='ASVspoof05 Dev Spoofed',
        title='PMF of bonafide vs spoof in the dev ASVspoof05 database'
    )

    # Eval
    plot_precomputed_pmfs(
        pmf_probs_bonafide_eval_ASVspoof05,
        pmf_probs_spoofed_eval_ASVspoof05,
        label_a='ASVspoof05 Eval Bonafide',
        label_b='ASVspoof05 Eval Spoofed',
        title='PMF of bonafide vs spoof in the eval ASVspoof05 database'
    )



    # === ASVspoof2019 ===

    # Train
    plot_precomputed_pmfs(
        pmf_probs_bonafide_train_ASVspoof2019,
        pmf_probs_spoofed_train_ASVspoof2019,
        label_a='ASVspoof2019 Train Bonafide',
        label_b='ASVspoof2019 Train Spoofed',
        title='PMF of bonafide vs spoof in the train ASVspoof2019 database'
    )

    # Dev
    plot_precomputed_pmfs(
        pmf_probs_bonafide_dev_ASVspoof2019,
        pmf_probs_spoofed_dev_ASVspoof2019,
        label_a='ASVspoof2019 Dev Bonafide',
        label_b='ASVspoof2019 Dev Spoofed',
        title='PMF of bonafide vs spoof in the dev ASVspoof2019 database'
    )

    # Eval
    plot_precomputed_pmfs(
        pmf_probs_bonafide_eval_ASVspoof2019,
        pmf_probs_spoofed_eval_ASVspoof2019,
        label_a='ASVspoof2019 Eval Bonafide',
        label_b='ASVspoof2019 Eval Spoofed',
        title='PMF of bonafide vs spoof in the eval ASVspoof2019 database'
    )

   

    


   