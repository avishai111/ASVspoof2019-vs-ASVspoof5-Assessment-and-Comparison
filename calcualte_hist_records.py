import os
import numpy as np
import soundfile as sf
from tqdm import tqdm


def process_asvspoof_dataset(audio_folder, metadata_file, output_prefix, num_bins=1000, label_column=5):
    """
    Process ASVspoof dataset to compute PMF histograms and save in .npy format.

    Args:
        audio_folder (str): Path to audio files (.flac).
        metadata_file (str): Path to metadata text file.
        output_prefix (str): Prefix for output .npy files.
        num_bins (int): Number of histogram bins.
        label_column (int): Index of the label column in metadata (0-based).
    """
    # Read metadata
    with open(metadata_file, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]

    file_names = [line[1] for line in lines]
    file_labels = [line[label_column] for line in lines]

    # Initialize
    pmf_counts_bonafide = np.zeros(num_bins)
    pmf_counts_spoofed = np.zeros(num_bins)
    bin_edges_bonafide = None
    bin_edges_spoofed = None
    total_samples_bonafide = 0
    total_samples_spoofed = 0
    flag_bonafide = False
    flag_spoofed = False

    for i in tqdm(range(len(file_names)), desc="Processing files"):
        file_path = os.path.join(audio_folder, file_names[i] + '.flac')

        if os.path.isfile(file_path):
            y, fs = sf.read(file_path)
            label = file_labels[i].lower()

            if label == 'bonafide':
                counts, bin_edges = np.histogram(y, bins=num_bins, range=(-1, 1))
                pmf_counts_bonafide += counts
                total_samples_bonafide += len(y)
                bin_edges_bonafide = bin_edges
                flag_bonafide = True

            elif label == 'spoof':
                counts, bin_edges = np.histogram(y, bins=num_bins, range=(-1, 1))
                pmf_counts_spoofed += counts
                total_samples_spoofed += len(y)
                bin_edges_spoofed = bin_edges
                flag_spoofed = True

            else:
                print(f"Error label for file: {file_names[i]}")

            if flag_bonafide and flag_spoofed:
                if not np.array_equal(bin_edges_bonafide, bin_edges_spoofed):
                    raise ValueError("Problem in bins number!")

        else:
            print(f"File {file_path} not found.")

    # Normalize to PMF
    pmf_probs_bonafide = pmf_counts_bonafide / total_samples_bonafide if total_samples_bonafide > 0 else np.zeros(num_bins)
    pmf_probs_spoofed = pmf_counts_spoofed / total_samples_spoofed if total_samples_spoofed > 0 else np.zeros(num_bins)

    # Save as .npy files
    np.save(f"{output_prefix}_pmf_probs_bonafide.npy", pmf_probs_bonafide)
    np.save(f"{output_prefix}_pmf_probs_spoofed.npy", pmf_probs_spoofed)
    np.save(f"{output_prefix}_bin_edges.npy", bin_edges_bonafide if flag_bonafide else bin_edges_spoofed)

    print(f"\nSaved .npy files with prefix: {output_prefix}")



if __name__ == "__main__":
    # Train
    process_asvspoof_dataset(
        audio_folder='E:/ASVSpoof5/flac_T/',
        metadata_file='E:/ASVSpoof5/cm_protocols/ASVspoof5.train.metadata.txt',
        output_prefix='train_data',
        label_column=5
    )

    # Dev
    process_asvspoof_dataset(
        audio_folder='E:/ASVSpoof5/flac_D/',
        metadata_file='E:/ASVSpoof5/cm_protocols/ASVspoof5.dev.metadata.txt',
        output_prefix='validation_data',
        label_column=5
    )

    # Eval
    process_asvspoof_dataset(
        audio_folder='E:/ASVSpoof5/flac_E/',
        metadata_file='E:/ASVSpoof5/cm_protocols/ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv',
        output_prefix='eval_data',
        label_column=8
    )
