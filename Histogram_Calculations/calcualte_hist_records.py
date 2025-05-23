import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import islice


def process_file(args):
    """
    Process a single audio file to compute histogram counts.
    """
    file_path, label, num_bins = args
    if not os.path.isfile(file_path):
        print(f"File {file_path} not found.")
        return None

    try:
        y, fs = sf.read(file_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    counts, bin_edges = np.histogram(y, bins=num_bins, range=(-1, 1))
    return label, counts.astype(np.uint64), bin_edges, len(y)


def chunks(iterable, size):
    """
    Yield successive chunks from iterable.
    """
    it = iter(iterable)
    for first in it:
        yield [first] + list(islice(it, size - 1))


def process_asvspoof_dataset_parallel(audio_folder, metadata_file, output_prefix, num_bins=2**16, label_column=5, max_workers=4, batch_size=10000):
    """
    Process ASVspoof dataset in parallel in batches to compute PMF histograms and save in .npy format.
    """
    with open(metadata_file, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]

    file_names = [line[1] for line in lines]
    file_labels = [line[label_column].lower() for line in lines]

    args = [(os.path.join(audio_folder, fname + '.flac'), label, num_bins) for fname, label in zip(file_names, file_labels)]

    pmf_counts_bonafide = np.zeros(num_bins, dtype=np.uint64)
    pmf_counts_spoofed = np.zeros(num_bins, dtype=np.uint64)
    total_samples_bonafide = 0
    total_samples_spoofed = 0
    bin_edges = None

    total_files = len(args)
    processed = 0

    for batch in chunks(args, batch_size):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(process_file, batch):
                processed += 1
                if result is None:
                    continue

                label, counts, bin_edges_local, n_samples = result
                if bin_edges is None:
                    bin_edges = bin_edges_local

                if label == 'bonafide':
                    pmf_counts_bonafide += counts
                    total_samples_bonafide += n_samples
                elif label == 'spoof':
                    pmf_counts_spoofed += counts
                    total_samples_spoofed += n_samples
                else:
                    print(f"Unknown label: {label}")

        tqdm.write(f"✅ Processed {processed}/{total_files} files...")

    # Normalize to PMF
    pmf_probs_bonafide = pmf_counts_bonafide / total_samples_bonafide if total_samples_bonafide > 0 else np.zeros(num_bins)
    pmf_probs_spoofed = pmf_counts_spoofed / total_samples_spoofed if total_samples_spoofed > 0 else np.zeros(num_bins)

    # Save as .npy files
    np.save(f"{output_prefix}_pmf_probs_bonafide.npy", pmf_probs_bonafide)
    np.save(f"{output_prefix}_pmf_probs_spoofed.npy", pmf_probs_spoofed)
    np.save(f"{output_prefix}_bin_edges.npy", bin_edges)

    print(f"\n✅ Saved .npy files with prefix: {output_prefix}")


if __name__ == "__main__":
    # #Train ASVspoof5
    # process_asvspoof_dataset_parallel(
    #     audio_folder='E:/ASVSpoof5/flac_T/',
    #     metadata_file='E:/ASVSpoof5/cm_protocols/ASVspoof5.train.metadata.txt',
    #     output_prefix='train_data_ASVSpoof5',
    #     label_column=5
    # )

    # Dev ASVspoof5
    process_asvspoof_dataset_parallel(
        audio_folder='E:/ASVSpoof5/flac_D/',
        metadata_file='E:/ASVSpoof5/cm_protocols/ASVspoof5.dev.metadata.txt',
        output_prefix='validation_data_ASVSpoof5',
        label_column=5
    )

    # Eval ASVspoof5
    process_asvspoof_dataset_parallel(
        audio_folder='E:/ASVSpoof5/flac_E/',
        metadata_file='E:/ASVSpoof5/cm_protocols/ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv',
        output_prefix='eval_data_ASVSpoof5',
        label_column=8
    )

    # Train ASVspoof2019
    process_asvspoof_dataset_parallel(
        audio_folder='C:/AsvSpoof/databases/2019/LA/ASVspoof2019_LA_train/flac/',
        metadata_file='C:/AsvSpoof/databases/2019/LA/ASVspoof2019_LA_cm_protocols/with_sex_labels/ASVspoof2019.LA.cm.train.trn.txt',
        output_prefix='train_data_ASVSpoof2019',
        label_column=5
    )

    # Dev ASVspoof2019
    process_asvspoof_dataset_parallel(
        audio_folder='C:/AsvSpoof/databases/2019/LA/ASVspoof2019_LA_dev/flac/',
        metadata_file='C:/AsvSpoof/databases/2019/LA/ASVspoof2019_LA_cm_protocols/with_sex_labels/ASVspoof2019.LA.cm.dev_old_witout_enroll.trl.txt',
        output_prefix='validation_data_ASVSpoof2019',
        label_column=5
    )

    # Eval ASVspoof2019
    process_asvspoof_dataset_parallel(
        audio_folder='C:/AsvSpoof/databases/2019/LA/ASVspoof2019_LA_eval/flac/',
        metadata_file='C:/AsvSpoof/databases/2019/LA/ASVspoof2019_LA_cm_protocols/with_sex_labels/ASVspoof2019.LA.cm.eval_old_witout_enroll.trl.txt',
        output_prefix='eval_data_ASVSpoof2019',
        label_column=5
    )