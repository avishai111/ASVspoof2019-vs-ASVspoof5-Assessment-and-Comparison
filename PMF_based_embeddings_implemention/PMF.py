import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from typing import List, Tuple
from GammatoneFilter import GammatoneFilterbank
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import convolve
import soundfile as sf
from tqdm import tqdm
from PMF_based_embeddings_implemention.CONSTANTS import NUM_BINS, HIST_EDGES

class PMF:
    def __init__(self, files_folder, protocol_file=None, ftype=None):
        """
        Parameters:
        -----------
        files_folder : str
            Path to the folder containing FLAC audio files.
        protocol_file : str or None
            Path to the protocol file that lists which file is spoofed vs. bonafide.
            Expected line format:
                <speaker_id> <file_id> <dummy_field> <attack_id> <label>
            For bonafide samples, the attack_id is '-' and the label is typically 'bonafide'.
            For spoofed samples, the attack_id is e.g. 'A01' and the label is 'spoof'.
        ftype : callable or None
            A function that filters the audio signal. It should accept a 1D numpy array (audio samples)
            and return either a filtered 1D array or a 2D array (if multi-channel filtering is applied).
            If None, no filtering is applied.
        """
        self.files_folder = files_folder
        self.ftype = ftype
        self.hist = None          # Will hold the overall histogram (aggregated over files)
        self.samples_in_hist = None
        self.hist_edges = None    # The bin edges used for the histogram
        
        # List of all FLAC files in the folder (if no protocol file is provided,
        # this will be used for overall histogram computation)
        self.file_list = self._get_flac_files(files_folder)
        
        # Additional fields for categorized samples:
        self.bonafide_files = []
        self.spoof_files = []
        self.attack_files = {}  # dictionary mapping attack ID (e.g., "A01") to list of files
        
        if protocol_file is not None:
            self._parse_protocol_file(protocol_file)
            
        print(f"Loaded {len(self.file_list)} FLAC files.")
        print(f"Number of Bonafide samples in protocol: {len(self.bonafide_files)}")
        print(f"Number of Spoofed samples in protocol: {len(self.spoof_files)}")
        print(f'Attack names in protocol: {sorted(list(self.attack_files.keys()))}')

    def _get_flac_files(self, folder):
        """Return a list of full paths to .flac files in the given folder."""
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith('.flac')]
        return files

    def _parse_protocol_file(self, protocol_file):
        """
        Parse the protocol file to categorize files into spoofed and bonafide.

        Expected line format:
            <speaker_id> <file_id> <irrelevant_field> <attack_id> <label>
        Example:
            LA_0086 LA_T_8004323 - A01 spoof
        Note:
            For bonafide samples the attack_id is '-' (and label is 'bonafide').
        """
        with open(protocol_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 5:
                    continue  # Skip malformed lines
                
                speaker_id = tokens[0]
                file_id = tokens[1]
                # tokens[2] is a dummy field (often '-')
                attack_id = tokens[3]
                label = tokens[4].lower()  # Expecting 'spoof' or 'bonafide'
                gender = tokens[5] if len(tokens) > 5 else None  # Optional field
                # Build the full file path (assuming file names are like "<file_id>.flac")
                file_path = os.path.join(self.files_folder, file_id + '.flac')
                
                # Categorize based on the attack field (or label)
                if attack_id == '-' or label == 'bonafide':
                    self.bonafide_files.append(file_path)
                else:
                    self.spoof_files.append(file_path)
                    # Also store by attack ID
                    if attack_id not in self.attack_files:
                        self.attack_files[attack_id] = []
                    self.attack_files[attack_id].append(file_path)


    def compute_hist_per_input_file_stream(self,file_path: str,num_bins: int = 512,hist_edges: Tuple[float, float] = (-1.0, 1.0)) -> Tuple[np.ndarray, str]:
        """
        Compute the PMF histogram for a single audio file.

        Parameters
        ----------
        file_path : str
            Path to the audio file.
        num_bins : int
            Number of bins for the histogram.
        hist_edges : Tuple[float, float]
            Range (min, max) for the histogram bins.

        Returns
        -------
        Tuple:
            - np.ndarray of shape (n_channels, num_bins): PMF(s) for the file
            - str: filename (basename of file_path)
        """
        if not os.path.isfile(file_path):
            print(f"❌ File not found: {file_path}")
            return np.array([]), ""

        scale = num_bins / (hist_edges[1] - hist_edges[0])
        audio, _ = sf.read(file_path, dtype='float32')

        if audio.ndim != 1:
            raise ValueError(f'{file_path} is not mono.')

        # Optional filtering
        sig = self.ftype.filter_signal(audio) if self.ftype else audio[None, :]
        n_channels = sig.shape[0]
        pmfs = []

        for ch in range(n_channels):
            idx = np.floor((sig[ch] - hist_edges[0]) * scale).astype(np.int32)
            np.clip(idx, 0, num_bins - 1, out=idx)
            hist = np.bincount(idx, minlength=num_bins)
            pmf = hist / hist.sum()
            pmfs.append(pmf)

        return np.array(pmfs), os.path.basename(file_path)  # shape: (n_channels, num_bins)

                    
    def compute_hist_per_file_stream(self, num_bins: int = 512, hist_edges: Tuple[float, float] = (-1.0, 1.0)) -> Tuple[np.ndarray, List[str]]:
        """
        Compute a separate PMF for each file and return results as a NumPy array with corresponding filenames.

        Returns
        -------
        Tuple:
            - np.ndarray of shape (n_files, n_channels, num_bins): PMFs per file
            - List[str]: filenames corresponding to each row in the array
        """
        
        if not self.file_list:
            print("No FLAC files found in the specified folder.")
            return np.array([]), []

        scale = num_bins / (hist_edges[1] - hist_edges[0])
        all_pmfs = []
        filenames = []

        for i, path in enumerate(self.file_list, 1):
            audio, _ = sf.read(path, dtype='float32')
            if audio.ndim != 1:
                raise ValueError(f'{path} is not mono.')

            sig = self.ftype.filter_signal(audio) if self.ftype else audio[None, :]
            n_channels = sig.shape[0]
            pmfs = []

            for ch in range(n_channels):
                idx = np.floor((sig[ch] - hist_edges[0]) * scale).astype(np.int32)
                np.clip(idx, 0, num_bins - 1, out=idx)
                hist = np.bincount(idx, minlength=num_bins)
                pmf = hist / hist.sum()
                pmfs.append(pmf)

            all_pmfs.append(pmfs)
            filenames.append(os.path.basename(path))

            if i % 1000 == 0:
                print(f"[{i}/{len(self.file_list)}] files processed")

        return np.array(all_pmfs), filenames  # shape: (n_files, n_channels, num_bins)




    def compute_hist_stream(self, num_bins: int = NUM_BINS, 
                            hist_edges: Tuple[float, float] = HIST_EDGES, 
                            edge_behavior: str = 'floor') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the histogram (PMF) of audio samples in all loaded files
        
        Returns
        -------
        List[Tuple[counts, pmf]]  -- one (hist, pmf) per filter channel
        """
        if not self.file_list:
            raise FileNotFoundError("No FLAC files found in the specified folder.")
            
        edges = np.linspace(hist_edges[0], hist_edges[1], num_bins + 1, dtype=np.float32)
        scale = num_bins / (hist_edges[1] - hist_edges[0])
        agg_hist = None
        for i, path in enumerate(tqdm(self.file_list, desc="Processing files"), 1):
            audio, _ = sf.read(path, dtype='float32')
            if audio.ndim != 1:
                raise ValueError(f'{path} is not mono.')

            # (n_filters, T)  or  (1, T) if no filter bank is used
            sig = (self.ftype.filter_signal(audio)
                if self.ftype is not None
                else audio[None, :])

            n_filters = sig.shape[0]
            if agg_hist is None:
                agg_hist = np.zeros((n_filters, num_bins), dtype=np.int64)

            # Choose binning method based on edge_behavior
            if edge_behavior == 'floor':
                idx = np.floor((sig - hist_edges[0]) * scale).astype(np.int32)
                np.clip(idx, 0, num_bins - 1, out=idx)
            elif edge_behavior == 'ceil':
                # Alternative: shift values slightly to change edge behavior
                epsilon = 1e-10 * (hist_edges[1] - hist_edges[0])
                idx = np.floor((sig - hist_edges[0] - epsilon) * scale).astype(np.int32)
                np.clip(idx, 0, num_bins - 1, out=idx)
            elif edge_behavior == 'digitize_right':
                idx = np.digitize(sig.flatten(), edges, right=True) - 1
                np.clip(idx, 0, num_bins - 1, out=idx)
                idx = idx.reshape(sig.shape)
            elif edge_behavior == 'digitize_left':
                idx = np.digitize(sig.flatten(), edges, right=False) - 1
                np.clip(idx, 0, num_bins - 1, out=idx)
                idx = idx.reshape(sig.shape)
            else:
                raise ValueError(f"Unknown edge_behavior: {edge_behavior}. Use 'floor', 'ceil', 'digitize_right', or 'digitize_left'.")

            offset_idx = idx + (np.arange(n_filters)[:, None] * num_bins)      # shape (n_filters, T)
            counts = np.bincount(
                offset_idx.ravel(),
                minlength=n_filters * num_bins
            ).reshape(n_filters, num_bins)

            agg_hist += counts

        # Normalise
        pmf = agg_hist / agg_hist.sum(axis=1, keepdims=True)
        return [(agg_hist[k], pmf[k]) for k in range(agg_hist.shape[0])]
    
    def compute_hist(self, num_bins: int = NUM_BINS, 
                     hist_edges: tuple = HIST_EDGES, 
                     edge_behavior: str = 'floor') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the overall histogram (PMF) of the audio samples in all loaded files.
        This method uses self.file_list.

        Parameters:
        ----------- 
        num_bins : int, optional
            Number of bins to use for the histogram.
        hist_edges : tuple, optional
            Tuple containing the minimum and maximum values for the histogram bins.
        edge_behavior : str, optional
            Controls how edge values are handled:
            - 'floor': upper edge values go to last bin (current behavior)
            - 'ceil': lower edge values go to first bin, upper edge excluded
            - 'digitize_right': right edges inclusive (numpy.digitize with right=True)
            - 'digitize_left': left edges inclusive (numpy.digitize with right=False)
            
        Returns:
        --------
        List[Tuple[counts, pmf]] -- one (hist, pmf) per filter channel
        """
        if not self.file_list:
            raise FileNotFoundError("No FLAC files in folder.")

        self.hist_edges = np.linspace(hist_edges[0], hist_edges[1], num_bins + 1)
        self.samples_in_hist = 0
        
        # Initialize aggregation variables
        agg_hist = None
        
        for i, file in enumerate(tqdm(self.file_list, desc="Processing files"), 1):
            audio, sr = sf.read(file, dtype='float32')
            if audio.ndim != 1:
                raise ValueError("Only mono audio files are supported.")
                
            if self.ftype is not None:
                # Apply filterbank: shape (n_filters, n_samples)
                filtered_signals = self.ftype.filter_signal(audio)
                self.samples_in_hist += filtered_signals.shape[1]
                
                # Initialize aggregated histogram on first file
                if agg_hist is None:
                    n_filters = filtered_signals.shape[0]
                    agg_hist = np.zeros((n_filters, num_bins), dtype=np.int64)
                
                # Efficiently bin samples using vectorized operations
                # Convert samples to bin indices
                scale = num_bins / (hist_edges[1] - hist_edges[0])
                
                # Choose binning method based on edge_behavior
                if edge_behavior == 'floor':
                    idx = np.floor((filtered_signals - hist_edges[0]) * scale).astype(np.int32)
                    np.clip(idx, 0, num_bins - 1, out=idx)
                elif edge_behavior == 'ceil':
                    # Alternative: shift values slightly to change edge behavior
                    epsilon = 1e-10 * (hist_edges[1] - hist_edges[0])
                    idx = np.floor((filtered_signals - hist_edges[0] - epsilon) * scale).astype(np.int32)
                    np.clip(idx, 0, num_bins - 1, out=idx)
                elif edge_behavior == 'digitize_right':
                    idx = np.digitize(filtered_signals.flatten(), self.hist_edges, right=True) - 1
                    np.clip(idx, 0, num_bins - 1, out=idx)
                    idx = idx.reshape(filtered_signals.shape)
                elif edge_behavior == 'digitize_left':
                    idx = np.digitize(filtered_signals.flatten(), self.hist_edges, right=False) - 1
                    np.clip(idx, 0, num_bins - 1, out=idx)
                    idx = idx.reshape(filtered_signals.shape)
                else:
                    raise ValueError(f"Unknown edge_behavior: {edge_behavior}. Use 'floor', 'ceil', 'digitize_right', or 'digitize_left'.")
                
                # Accumulate histogram counts per filter channel
                for ch in range(filtered_signals.shape[0]):
                    np.add.at(agg_hist[ch], idx[ch], 1)
                    
            else:
                # No filtering case
                self.samples_in_hist += audio.shape[0]
                
                if agg_hist is None:
                    agg_hist = np.zeros((1, num_bins), dtype=np.int64)
                
                # Bin the raw audio
                scale = num_bins / (hist_edges[1] - hist_edges[0])
                
                # Choose binning method based on edge_behavior
                if edge_behavior == 'floor':
                    idx = np.floor((audio - hist_edges[0]) * scale).astype(np.int32)
                    np.clip(idx, 0, num_bins - 1, out=idx)
                elif edge_behavior == 'ceil':
                    # Alternative: shift values slightly to change edge behavior
                    epsilon = 1e-10 * (hist_edges[1] - hist_edges[0])
                    idx = np.floor((audio - hist_edges[0] - epsilon) * scale).astype(np.int32)
                    np.clip(idx, 0, num_bins - 1, out=idx)
                elif edge_behavior == 'digitize_right':
                    idx = np.digitize(audio, self.hist_edges, right=True) - 1
                    np.clip(idx, 0, num_bins - 1, out=idx)
                elif edge_behavior == 'digitize_left':
                    idx = np.digitize(audio, self.hist_edges, right=False) - 1
                    np.clip(idx, 0, num_bins - 1, out=idx)
                else:
                    raise ValueError(f"Unknown edge_behavior: {edge_behavior}. Use 'floor', 'ceil', 'digitize_right', or 'digitize_left'.")
                
                np.add.at(agg_hist[0], idx, 1)
            
            if i % 1000 == 0 and i > 0:
                gc.collect()
        
        # Normalize to get PMFs (handle potential division by zero)
        hist_sums = agg_hist.sum(axis=1, keepdims=True)
        agg_pmfs = np.where(hist_sums > 0, agg_hist / hist_sums, 0)
        
        # Return in the expected format
        return [(agg_hist[i], agg_pmfs[i]) for i in range(agg_hist.shape[0])]

    def compute_hist_by_category(self, 
                                 category: str, 
                                 num_bins: int = NUM_BINS, 
                                 hist_edges: tuple = HIST_EDGES) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the normalized histogram (PMF) for a specific category of files.
        NOTE: This method is deprecated. Use compute_hist_by_category_stream instead for better performance.

        Parameters:
        -----------
        category : str
            Category for which to compute the histogram. Acceptable values:
              - "spoof": all spoofed samples
              - "bonafide": all bonafide samples
              - or a specific attack ID (e.g., "A01") for spoofed samples of that attack.
        num_bins : int, optional
            Number of bins to use for the histogram.
        hist_edges : tuple, optional
            Tuple containing the minimum and maximum values for the histogram bins.
            
        Returns:
        --------
        If no filterbank: A tuple (pmf, edges) where pmf is 1D array
        If filterbank: A tuple (result, edges) where result is list of (hist, pmf) tuples per filter
        """
        print("WARNING: compute_hist_by_category is deprecated. Use compute_hist_by_category_stream for better performance.")
        
        # Choose the appropriate file list based on the category
        if category.lower() == 'spoof':
            files = self.spoof_files
        elif category.lower() == 'bonafide':
            files = self.bonafide_files
        elif category in self.attack_files:
            files = self.attack_files[category]
        else:
            raise ValueError("Category not recognized. Use 'spoof', 'bonafide', or a valid attack ID.")

        if not self.file_list:
            raise FileNotFoundError("No FLAC files in folder.")
        
        edges = np.linspace(hist_edges[0], hist_edges[1], num_bins + 1)
        
        if self.ftype is None:
            # No filterbank - original behavior
            all_samples = []
            for file in files:
                audio, sr = sf.read(file, dtype='float32')
                if audio.ndim != 1:
                    raise ValueError("Only mono audio files are supported.")
                all_samples.append(audio)
            
            all_samples_concat = np.concatenate(all_samples)
            hist, _ = np.histogram(all_samples_concat, bins=edges)
            pmf = hist / np.sum(hist)
            return pmf, edges
        else:
            # With filterbank - use streaming approach
            result = self.compute_hist_by_category_stream(category, num_bins, hist_edges)
            if result is None:
                return None
            return result

    def compute_hist_by_category_stream(self, 
                                        category: str, 
                                        num_bins: int = NUM_BINS, 
                                        hist_edges: tuple = HIST_EDGES, 
                                        edge_behavior: str = 'floor'):
        """
        Compute the histogram (PMF) for a specific category of files

        Parameters
        ----------
        category : str
            'spoof', 'bonafide', or a specific attack-ID (e.g. 'A01').
        num_bins : int, optional
            Number of histogram bins.
        hist_edges : tuple, optional
            (min, max) bounds of the histogram.
        edge_behavior : str, optional
            Controls how edge values are handled:
            - 'floor': upper edge values go to last bin (current behavior)
            - 'ceil': lower edge values go to first bin, upper edge excluded
            - 'digitize_right': right edges inclusive (numpy.digitize with right=True)
            - 'digitize_left': left edges inclusive (numpy.digitize with right=False)

        Returns
        -------
        If no filter bank is used (self.ftype is None):
            pmf : np.ndarray(shape=(num_bins,))
            edges : np.ndarray(shape=(num_bins+1,))
        If a filter bank is used:
            result : list[tuple[counts, pmf]]  – one pair per filter channel
            edges  : np.ndarray(shape=(num_bins+1,))
        """
        if category.lower() == "spoof":
            files = self.spoof_files
        elif category.lower() == "bonafide":
            files = self.bonafide_files
        elif category in self.attack_files:
            files = self.attack_files[category]
        else:
            raise ValueError("Category not recognized. Use 'spoof', 'bonafide', or a valid attack ID.")

        if not self.file_list:
            raise FileNotFoundError("No FLAC files in folder.")

        edges = np.linspace(hist_edges[0], hist_edges[1], num_bins + 1, dtype=np.float32)
        scale = num_bins / (hist_edges[1] - hist_edges[0])
        agg_hist = None

        for i, path in tqdm(enumerate(files, 1), desc=f"Processing {category} files",total=len(files)):
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim != 1:
                raise ValueError(f"{path} is not mono.")

            # Apply filter bank if requested
            sig = (self.ftype.filter_signal(audio) if self.ftype is not None else audio[None, :])

            n_chan = sig.shape[0]
            if agg_hist is None:
                agg_hist = np.zeros((n_chan, num_bins), dtype=np.int64)

            # Choose binning method based on edge_behavior
            if edge_behavior == 'floor':
                # Current behavior: upper edge goes to last bin
                idx = np.floor((sig - hist_edges[0]) * scale).astype(np.int32)
                np.clip(idx, 0, num_bins - 1, out=idx)
            elif edge_behavior == 'ceil':
                # Alternative: shift values slightly to change edge behavior
                # This makes edge values go to the previous bin
                epsilon = 1e-10 * (hist_edges[1] - hist_edges[0])
                idx = np.floor((sig - hist_edges[0] - epsilon) * scale).astype(np.int32)
                np.clip(idx, 0, num_bins - 1, out=idx)
            elif edge_behavior == 'digitize_right':
                # Right edges inclusive: [a, b], (b, c], (c, d], ..., (y, z]
                idx = np.digitize(sig.flatten(), edges, right=True) - 1
                np.clip(idx, 0, num_bins - 1, out=idx)
                idx = idx.reshape(sig.shape)
            elif edge_behavior == 'digitize_left':
                # Left edges inclusive: [a, b), [b, c), [c, d), ..., [y, z)
                idx = np.digitize(sig.flatten(), edges, right=False) - 1
                np.clip(idx, 0, num_bins - 1, out=idx)
                idx = idx.reshape(sig.shape)
            else:
                raise ValueError(f"Unknown edge_behavior: {edge_behavior}. Use 'floor', 'ceil', 'digitize_right', or 'digitize_left'.")

            # Accumulate per channel
            for ch in range(n_chan):
                np.add.at(agg_hist[ch], idx[ch], 1)

            if i % 1_000 == 0:
                print(f"[{i}/{len(files)}] files processed for category '{category}'")

        if self.ftype is None: # No filter bank used
            pmf = agg_hist[0] / agg_hist[0].sum()
            return pmf, edges
        else:
            # For filterbank case, return PMF per filter channel (like MATLAB)
            # Each filter gets its own normalized PMF
            pmf_list = []
            for ch in range(n_chan):
                if agg_hist[ch].sum() > 0:
                    pmf = agg_hist[ch] / agg_hist[ch].sum()
                else:
                    pmf = np.zeros(num_bins, dtype=np.float32)
                pmf_list.append(pmf)
            return pmf_list, edges

    def plot(self):
        """
        Plot the overall histogram/PMF computed by compute_hist().
        """
        if self.hist is None:
            print("No histogram computed. Call 'compute_hist()' first.")
            return
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.hist, bins=self.hist_edges, kde=False)
        plt.title("Probability Mass Function (PMF) of the Audio Samples (All Files)")
        plt.xlabel("Sample Value")
        plt.ylabel("Count")
        plt.show()

    def plot_by_category(self, 
                         category: str, 
                         num_bins: int = NUM_BINS, 
                         hist_edges: tuple = HIST_EDGES):
        """
        Plot the PMF for a specific category.

        Parameters:
        -----------
        category : str
            Category to plot. Acceptable values: 'spoof', 'bonafide', or an attack ID (e.g., 'A01').
        num_bins : int, optional
            Number of bins to use for the histogram.
        hist_edges : tuple, optional
            Tuple containing the minimum and maximum values for the histogram bins.
        """
        result = self.compute_hist_by_category(category, num_bins, hist_edges)
        if result is None:
            return
        pmf, edges = result
        centers = (edges[:-1] + edges[1:]) / 2
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(centers, pmf, step='pre', color='orange', alpha=0.4)
        plt.title(f"PMF of Audio Samples for Category: {category}")
        plt.xlabel("Sample Value")
        plt.ylabel("Normalized Frequency")
        plt.xlim(-0.01, 0.01)
        plt.ylim(0, max(17e-3, np.max(pmf) * 1.1))
        plt.show()
        
        

if __name__ == "__main__":
    # Example usage
    files_folder = 'C:/Users/avish/OneDrive/Desktop/avishai111-ASVspoof2019-vs-ASVspoof5-Assessment-and-Comparison/PMF_based_embeddings_implemention/LA/ASVspoof2019_LA_train/flac/'  # Replace with your actual folder path
    protocol_file = 'C:/Users/avish/OneDrive/Desktop/avishai111-ASVspoof2019-vs-ASVspoof5-Assessment-and-Comparison/PMF_based_embeddings_implemention/ASVspoof2019/ASVspoof2019.LA.cm.train.trn.txt'
    num_filters = 10
    sample_rate = 16000
    low_freq = np.finfo(float).eps
    high_freq = sample_rate // 2
    num_fft = 512
    
    gfb = GammatoneFilterbank(
    num_filters=num_filters,
    sample_rate=sample_rate,
    low_freq=low_freq,
    high_freq=high_freq,
    num_fft=num_fft,
    with_inverse=True
    )
    
    pmf = PMF(files_folder=files_folder, protocol_file=protocol_file, ftype=gfb)  # Replace with your actual protocol file path if needed
    pmf.compute_hist(num_bins=2**16, hist_edges=(-1.0, 1.0))
    pmf.plot()
    
    pmf.plot_by_category('spoof', num_bins=512, hist_edges=(-1.0, 1.0))
    pmf.plot_by_category('bonafide', num_bins=512, hist_edges=(-1.0, 1.0))
    pmf.plot_by_category('A01', num_bins=512, hist_edges=(-1.0, 1.0))  # Example attack ID
    
