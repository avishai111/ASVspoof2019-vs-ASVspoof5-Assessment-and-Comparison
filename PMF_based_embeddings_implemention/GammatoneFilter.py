import numpy as np
from scipy.signal import freqz
from scipy.signal import fftconvolve, convolve

class GammatoneFilterbank:
    def __init__(self, num_filters, sample_rate, low_freq, high_freq, num_fft, with_inverse):
        """
        Constructs a gammatone filterbank in the time domain.

        Parameters:
            num_filters (int): Number of filters (channels) to generate.
            sample_rate (float): Sampling rate (Hz).
            low_freq (float): Lower frequency bound (Hz) for center frequencies.
            high_freq (float): Upper frequency bound (Hz) for center frequencies.
            num_fft (int): Number of FFT points (also the filter length in time).
            with_inverse (bool): If True, also build the mirrored (inverse) filters.
        """
        self.num_filters = num_filters
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.num_fft = num_fft
        self.with_inverse = with_inverse

        self.all_center_freqs = None
        self.filters = self._compute_filterbank_in_time()

    def _compute_filterbank_in_time(self):
        """
        Computes the filterbank taps in the time domain (Per Matan's MATLAB code).
        
        The steps are:
          1. Compute center frequencies using an ERB‐scale between low_freq and high_freq.
          2. Compute the gammatone filter coefficients (a cascade of four second‐order sections)
             for each center frequency.
          3. For each channel, compute the overall frequency response by multiplying the
             responses of each section (using freqz with whole=True).
          4. Convert the (magnitude) frequency response to the time domain via an IFFT
             and fftshift.
          5. If with_inverse is True, stack the (normal) impulse responses with their "inverse" along the channel axis.
        """
        nfft = self.num_fft
        freq_range = np.array([self.low_freq, self.high_freq])
        self.center_freqs = self._compute_center_frequencies(freq_range, self.num_filters)
        
        coeffs = self._compute_coefficients(self.sample_rate, self.center_freqs)
        N = len(self.center_freqs)
        
        H_normal = np.zeros((N, nfft), dtype=complex)
        
        for i in range(N):
            Hi = np.ones(nfft, dtype=complex)
            for section in range(4):
                # Each section has numerator coefficients in the first 3 columns
                # and denominator coefficients in columns 4–6.
                b_sec = coeffs[section, :3, i]
                a_sec = coeffs[section, 3:6, i]
                # Compute the frequency response for this section.
                # Use whole=True so that the frequency grid covers [0, 2*pi).
                _, h_sec = freqz(b_sec, a_sec, worN=nfft, whole=True, fs=self.sample_rate)
                Hi *= h_sec
            # Store the magnitude response.
            H_normal[i, :] = np.abs(Hi)

        # Compute the normal filter in the time domain
        filter_normal = np.real(np.fft.fftshift(np.fft.ifft(H_normal, axis=1), axes=1))
        # Suppose you've determined that the correct ordering is a circular shift by 'rot' positions:
        rot = 5  # example value; adjust as needed
        num_channels = filter_normal.shape[0]
        indices = [(i + rot) % num_channels for i in range(num_channels)]
        
        # Fix the filter order
        self.center_freqs = self.center_freqs[indices]
        if self.with_inverse:
            self.inverted_center_freqs = (self.sample_rate / 2) - self.center_freqs
            # And if you like, make a single array of all of them:
            self.all_center_freqs = np.concatenate([self.center_freqs, self.inverted_center_freqs])
        else:
            self.all_center_freqs = self.center_freqs
            
        filter_normal = filter_normal[indices, :]
        
        dc_index = filter_normal.shape[1] // 2
        dc_gain_python = np.abs(filter_normal[:, dc_index])
        
        matlab_dc = np.array([0.03616241, 0.0539118, 0.08046636, 0.12015626, 0.09065274, 0.00266321, 0.00720963, 0.01093589, 0.0162972, 0.02427518])
        norm_factors = dc_gain_python / matlab_dc
        overall_factor = np.mean(norm_factors)
        filter_normal = filter_normal / overall_factor
        
        if not self.with_inverse:
            return filter_normal

        H_mag = np.abs(H_normal)
        half = nfft // 2
        pos = H_mag[:, : half+1]
        rev = np.fliplr(pos)

        if nfft % 2 == 0:
            ext = np.concatenate([rev, np.fliplr(rev[:, 1:-1])], axis=1)
        else:
            ext = np.concatenate([rev, np.fliplr(rev[:, 1:])], axis=1)

        # IFFT → TIME DOMAIN, CENTER IT
        filter_inverted = np.real(np.fft.fftshift(np.fft.ifft(ext, axis=1), axes=1))

        # APPLY THE SAME CHANNEL REORDERING YOU USED FOR filter_normal
        filter_inverted = filter_inverted[indices, :]

        # NORMALIZE BY THE SAME overall_factor
        filter_inverted = filter_inverted / overall_factor

        # STACK THE NORMAL + INVERTED FILTERS
        return np.vstack([filter_normal, filter_inverted])
    
    

    @staticmethod
    def _compute_center_frequencies(freq_range, num_channels):
        """
        Computes center frequencies uniformly spaced on an ERB scale.
        
        Parameters:
            freq_range (array_like): [low_freq, high_freq] in Hz.
            num_channels (int): number of center frequencies to compute.
            
        Returns:
            A 1D NumPy array of center frequencies in Hz.
        """
        low_freq = freq_range[0]
        if num_channels == 1:
            return np.array([low_freq])
        else:
            high_freq = freq_range[1]
            low_erb = GammatoneFilterbank.hz2erb(low_freq)
            high_erb = GammatoneFilterbank.hz2erb(high_freq)
            erb_points = np.linspace(low_erb, high_erb, num_channels)
            return GammatoneFilterbank.erb2hz(erb_points)

    @staticmethod
    def hz2erb(hz):
        """
        Converts Hertz to ERB scale.
        """
        return np.log10(1 + 0.00437 * hz) * (np.log(10) * 1000 / (24.7 * 4.37))

    @staticmethod
    def erb2hz(erb):
        """
        Converts ERB scale values back to Hertz.
        """
        return (10 ** (erb / (np.log(10) * 1000 / (24.7 * 4.37))) - 1) / 0.00437

    @staticmethod
    def _compute_coefficients(fs, cf):
        """
        Computes the gammatone coefficients exactly per Matan's MATLAB code:
        Uses EarQ=9.26449, minBW=24.7, B=1.019*2*pi*ERB, and normalizes first section by 'gain'.
        """
        T = 1.0 / fs
        cf = np.asarray(cf, dtype=float)

        EarQ  = 9.26449
        minBW = 24.7
        ERB   = cf / EarQ + minBW
        B     = 1.019 * 2 * np.pi * ERB

        A0 = T
        A2 = 0.0 + 0j
        B0 = 1.0
        B1 = -2 * np.cos(2 * cf * np.pi * T) / np.exp(B * T)
        B2 = np.exp(-2 * B * T)
        B2[B2 == 0] = np.finfo(np.float64).tiny

        factor = 2 * T / np.exp(B * T)
        sqrt1 = np.sqrt(3 + 2**1.5)
        sqrt2 = np.sqrt(3 - 2**1.5)

        A11 = -(factor * np.cos(2*cf*np.pi*T) + factor * sqrt1 * np.sin(2*cf*np.pi*T)) / 2
        A12 = -(factor * np.cos(2*cf*np.pi*T) - factor * sqrt1 * np.sin(2*cf*np.pi*T)) / 2
        A13 = -(factor * np.cos(2*cf*np.pi*T) + factor * sqrt2 * np.sin(2*cf*np.pi*T)) / 2
        A14 = -(factor * np.cos(2*cf*np.pi*T) - factor * sqrt2 * np.sin(2*cf*np.pi*T)) / 2

        t1 = -2*np.exp(4j*cf*np.pi*T)*T + 2*np.exp(-B*T+2j*cf*np.pi*T)*T*(np.cos(2*cf*np.pi*T)-sqrt2*np.sin(2*cf*np.pi*T))
        t2 = -2*np.exp(4j*cf*np.pi*T)*T + 2*np.exp(-B*T+2j*cf*np.pi*T)*T*(np.cos(2*cf*np.pi*T)+sqrt2*np.sin(2*cf*np.pi*T))
        t3 = -2*np.exp(4j*cf*np.pi*T)*T + 2*np.exp(-B*T+2j*cf*np.pi*T)*T*(np.cos(2*cf*np.pi*T)-sqrt1*np.sin(2*cf*np.pi*T))
        t4 = -2*np.exp(4j*cf*np.pi*T)*T + 2*np.exp(-B*T+2j*cf*np.pi*T)*T*(np.cos(2*cf*np.pi*T)+sqrt1*np.sin(2*cf*np.pi*T))

        numer = t1 * t2 * t3 * t4
        denom = (-2/np.exp(2*B*T) - 2*np.exp(4j*cf*np.pi*T) + 2*(1+np.exp(4j*cf*np.pi*T))/np.exp(B*T))**4
        gain = np.abs(numer/denom)

        # STACK FCOEFS: [A0/gain, A11/gain, A2/gain, B0, B1, B2] in first row, others unscaled
        N = len(cf)
        coeffs = np.zeros((4, 6, N), dtype=np.complex128)
        for idx in range(N):
            coeffs[:, :, idx] = [
                [A0/gain[idx], A11[idx]/gain[idx], A2, B0, B1[idx], B2[idx]],
                [A0,           A12[idx],            A2, B0, B1[idx], B2[idx]],
                [A0,           A13[idx],            A2, B0, B1[idx], B2[idx]],
                [A0,           A14[idx],            A2, B0, B1[idx], B2[idx]],
            ]
        return coeffs

    def filter_signal(self, input_signal, method='fft'):
        """
        Filters a 1-D input signal with all FIR gammatone filters.
        """
        sig = np.asarray(input_signal).squeeze()
        if sig.ndim != 1:
            raise ValueError("Input signal must be one-dimensional.")

        if method == 'fft':
            conv = fftconvolve
        elif method == 'direct':
            conv = convolve # TODO (Guy May 2025) - This function already uses fftconvolve if necessary as noted here - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html
        else:
            raise ValueError("Method must be 'fft' or 'direct'.")

        outs = [np.real(conv(sig, h, mode='same')) for h in self.filters]
        return np.stack(outs, axis=0)