import numpy as np
import pandas as pd
import scipy.io as sio
from PmfDist import PmfDist  # You will need to implement or port these functions

# Constants
num_bins = 2**16
bin_edges = np.linspace(-1, 1, num_bins)



def compute_pmf_distances(pmfs: dict, distance_module, output_filename=None):
    """
    Compute distance metrics for PMF pairs using specified distance functions.

    Args:
        pmfs (dict): A dictionary with keys as labels and values as (pmf1, pmf2) tuples.
        distance_module (class): A class containing static distance methods.
        output_filename (str, optional): Path to save the result as an Excel file.

    Returns:
        pd.DataFrame: A DataFrame with computed distances.
    """
    
    # Define distance functions
    distance_types = {
        'Modified Kolmogorov-Smirnov': distance_module.ks2_variant,
        'Kullback-Leibler': distance_module.kl_div,
        'Kullback-Leibler distance': distance_module.kl_dist,
        'Jensen-Shannon divergence': distance_module.js_div,
        'chi square': distance_module.chi_sqr,
        'Histogram Intersection': distance_module.hist_intersection,
        'Hellinger': distance_module.hellinger,
        'correlation': distance_module.corr,
    }

    # Compute distances
    results = []
    labels = list(pmfs.keys())
    for name, func in distance_types.items():
        row = [name]
        for key in labels:
            p1, p2 = pmfs[key]
            row.append(func(p1, p2))
        results.append(row)

    # Build column names
    columns = ['DistanceType'] + labels

    # Create DataFrame
    df = pd.DataFrame(results, columns=columns)

    # Save to Excel if requested
    if output_filename:
        df.to_excel(output_filename, index=False)
        print(f"Table saved to {output_filename}")

    return df


if __name__ == "__main__":

    # Load PMF data from .npy files
    pmfs = {
        'train_2019': (
            np.load('train_data_pmf_probs_bonafide.npy'),
            np.load('train_data_pmf_probs_spoofed.npy')
        ),
        'validation_2019': (
            np.load('validation_data_pmf_probs_bonafide.npy'),
            np.load('validation_data_pmf_probs_spoofed.npy')
        ),
        'eval_2019': (
            np.load('eval_data_pmf_probs_bonafide.npy'),
            np.load('eval_data_pmf_probs_spoofed.npy')
        ),
        'train_5': (
            np.load('ASVspoof5_train_data_pmf_probs_bonafide.npy'),
            np.load('ASVspoof5_train_data_pmf_probs_spoofed.npy')
        ),
        'validation_5': (
            np.load('ASVspoof5_validation_data_pmf_probs_bonafide.npy'),
            np.load('ASVspoof5_validation_data_pmf_probs_spoofed.npy')
        ),
        'eval_5': (
            np.load('ASVspoof5_eval_data_pmf_probs_bonafide.npy'),
            np.load('ASVspoof5_eval_data_pmf_probs_spoofed.npy')
        ),
    }

    # Compute distances and save to Excel
    df = compute_pmf_distances(pmfs, PmfDist, output_filename='distance_output.xlsx')
