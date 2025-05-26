import numpy as np

def chi_square_test(hist1, hist2):
    """
    Perform a Chi-Square test comparing two histogram-based distributions.
    
    Depending on the mode, this function can perform:
    
    Compare an observed histogram to an expected histogram (from theory or a model).  
    Optionally, the expected histogram can be scaled so that its total matches the observed total.
    
    Parameters
    ----------
    hist1 : array-like
    hist2 : array-like
    
    Returns
    -------
    result : float
          - 'chi2_stat': the computed chi-square statistic.
    
    Raises
    ------
    ValueError
        If the input histograms are not one-dimensional, have different lengths (when required),
        contain negative counts, or if an expected count is zero when the observed is not.
    """
    # Convert inputs to numpy arrays (and force them to be of type float)
    try:
        arr1 = np.asarray(hist1, dtype=np.float64)
        arr2 = np.asarray(hist2, dtype=np.float64)
    except Exception as e:
        raise ValueError("Could not convert inputs to numpy arrays.") from e

    # Check that the arrays are one-dimensional
    if arr1.ndim != 1 or arr2.ndim != 1:
        raise ValueError("Both input histograms must be one-dimensional arrays.")
    
    # Check that counts are nonnegative
    if np.any(arr1 < 0) or np.any(arr2 < 0):
        raise ValueError("Histogram counts must be nonnegative.")
    
    if arr1.size != arr2.size:
        raise ValueError("For a goodness-of-fit test, the observed and expected histograms must have the same number of bins.")
        
    # Remove bins where both observed and expected counts are zero
    # (these do not contribute to the chi-square statistic)
    valid_bins = (arr1 > 0) | (arr2 > 0)
    arr1 = arr1[valid_bins]
    arr2 = arr2[valid_bins]
        
    assert arr1.size == arr2.size and arr1.size > 0, "Histograms must have the same number of bins."
    # Compute the chi-square statistic manually.
    chi2_stat = 0.0
    valid_bins = 0
    for i, (obs, exp) in enumerate(zip(arr1, arr2)):
        chi2_stat += (obs - exp) ** 2 / (obs + exp) # ! Denominator should be exp + obs per Matan's thesis (cite 42), but it's exp only in the original expression.
        valid_bins += 1
    
    if valid_bins < 1:
        raise ValueError("No valid bins available to compute the chi-square statistic.")
            
    return chi2_stat


def correlation_distance(hist1, hist2):
    """
    Compute the correlation-based distance between two histograms using the Pearson correlation coefficient.
    
    The Pearson correlation coefficient, ρ, is computed by first subtracting the mean from each histogram:
    
        p' = p - mean(p)
        q' = q - mean(q)
    
    and then computing:
    
        ρ = sum(p' * q') / sqrt(sum(p'^2) * sum(q'^2))
    
    The correlation distance is then defined as:
    
        d = 1 - ρ
    
    Parameters
    ----------
    hist1 : array-like
        The first histogram as a one-dimensional array.
    hist2 : array-like
        The second histogram as a one-dimensional array.
    
    Returns
    -------
    result : float
          - 'distance': the computed correlation distance (1 - rho).
    
    Raises
    ------
    ValueError
        If the input histograms cannot be converted to one-dimensional numpy arrays,
        if they have different lengths, or if the denominator in the computation is zero
        (which typically indicates one of the histograms is constant).
    
    Examples
    --------
    >>> hist1 = [10, 20, 30, 40, 50]
    >>> hist2 = [12, 18, 33, 39, 52]
    >>> result = correlation_distance(hist1, hist2)
    """
    # Convert inputs to numpy arrays (ensuring they are floats)
    try:
        p = np.asarray(hist1, dtype=np.float64)
        q = np.asarray(hist2, dtype=np.float64)
    except Exception as e:
        raise ValueError("Could not convert the input histograms to numpy arrays.") from e

    # Check that both arrays are one-dimensional
    if p.ndim != 1 or q.ndim != 1:
        raise ValueError("Both histograms must be one-dimensional arrays.")

    # Check that the histograms have the same number of bins
    if p.size != q.size:
        raise ValueError("Histograms must have the same number of bins.")

    # Compute the means of both histograms
    p_mean = np.mean(p)
    q_mean = np.mean(q)

    # Compute mean-subtracted histograms (p' and q')
    p_prime = p - p_mean
    q_prime = q - q_mean

    # Compute the numerator and denominator for the Pearson correlation coefficient
    numerator = np.sum(p_prime * q_prime)
    denominator = np.sqrt(np.sum(p_prime**2) * np.sum(q_prime**2))

    # Check that the denominator is not zero (which would indicate constant input values)
    if denominator == 0:
        raise ValueError("Denominator is zero. One of the histograms may be constant, preventing computation of the correlation.")

    # Compute Pearson's correlation coefficient
    rho = numerator / denominator
    distance = 1 - rho  # Transform the correlation coefficient into a distance measure

    return distance


def hellinger_distance(p, q):
    """
    Compute the Hellinger distance between two probability distributions.
    
    The Hellinger distance is defined as:
    
        d_H = sqrt( 1 - sum_n sqrt( p(n) * q(n) ) )
    
    where p(n) and q(n) are the probability values for the nth bin of distributions p and q.
    
    Parameters
    ----------
    p : array-like
        The first distribution as a one-dimensional array of non-negative numbers.
    q : array-like
        The second distribution as a one-dimensional array of non-negative numbers.
    
    Returns
    -------
    result : float
        - 'hellinger_distance': the computed Hellinger distance.
    
    Raises
    ------
    ValueError
        If the inputs cannot be converted to one-dimensional numpy arrays,
        if they have different lengths, contain negative values.
    
    Examples
    --------
    >>> # With probabilities (already normalized)
    >>> p = [0.1, 0.4, 0.5]
    >>> q = [0.2, 0.3, 0.5]
    >>> result = hellinger_distance(p, q)
    """
    # Convert inputs to numpy arrays (as float)
    try:
        p_arr = np.asarray(p, dtype=float)
        q_arr = np.asarray(q, dtype=float)
    except Exception as e:
        raise ValueError("Could not convert inputs to numpy arrays.") from e
    
    # Ensure both arrays are one-dimensional.
    if p_arr.ndim != 1 or q_arr.ndim != 1:
        raise ValueError("Both input distributions must be one-dimensional arrays.")
    
    # Ensure both arrays have the same length.
    if p_arr.size != q_arr.size:
        raise ValueError("Both input distributions must have the same number of elements.")
    
    # Check that the distributions contain non-negative values.
    if np.any(p_arr < 0) or np.any(q_arr < 0):
        raise ValueError("Distributions must contain non-negative values only.")
        
    # Compute the sum of square roots of the products.
    similarity = np.sum(np.sqrt(p_arr * q_arr))
    
    # The Hellinger distance is the square root of (1 - similarity).
    # Note: For normalized distributions, similarity should lie between 0 and 1.
    if similarity < 0 or similarity > 1:
        print("Warning: Similarity value is out of expected bounds [0, 1]. Check normalization of distributions.")
    
    hellinger_dist = np.sqrt(1 - similarity)
    
    return hellinger_dist


def intersection_distance(p, q):
    """
    Compute the intersection-based distance between two distributions.
    
    The intersection measure is defined as:
    
        I = sum_n min(p(n), q(n))
    
    For probability mass functions (PMFs), I is in the range [0, 1]. To transform
    this similarity measure into a distance measure, we use:
    
        d = 1 - I
    
    Parameters
    ----------
    p : array-like
        The first distribution as a one-dimensional array of nonnegative numbers.
    q : array-like
        The second distribution as a one-dimensional array of nonnegative numbers.
    
    Returns
    -------
    result : float
        - 'distance': the computed distance d = 1 - I.
    
    Raises
    ------
    ValueError
        If the inputs cannot be converted to one-dimensional numpy arrays,
        if they have different lengths, or if they contain negative values.
    
    Examples
    --------
    >>> # Example 1: Using PMFs (already normalized)
    >>> p = [0.2, 0.5, 0.3]
    >>> q = [0.3, 0.4, 0.3]
    >>> result = intersection_distance(p, q)
    """
    # Convert inputs to numpy arrays as floats
    try:
        p_arr = np.asarray(p, dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64)
    except Exception as e:
        raise ValueError("Could not convert the inputs to numpy arrays.") from e

    # Ensure that both arrays are one-dimensional
    if p_arr.ndim != 1 or q_arr.ndim != 1:
        raise ValueError("Both input distributions must be one-dimensional arrays.")

    # Ensure both arrays have the same number of elements
    if p_arr.size != q_arr.size:
        raise ValueError("Both input distributions must have the same number of elements.")

    # Check that there are no negative values
    if np.any(p_arr < 0) or np.any(q_arr < 0):
        raise ValueError("Input distributions must contain nonnegative values only.")

    # Compute the intersection measure I
    intersection = np.sum(np.minimum(p_arr, q_arr))

    # Transform the similarity measure into a distance
    distance = 1 - intersection

    return distance


def abs_discount_smoothing(p, q, is_remove_common_zeros=True):
    """
    Apply absolute discount smoothing to two probability distributions.
    
    This function replicates the MATLAB absolute discount smoothing logic:
      - It creates masks for bins that are zero in one distribution but nonzero in the other.
      - It computes a small epsilon value as 0.001 times the minimum nonzero value from both distributions.
      - It computes a discount factor for the nonzero bins and subtracts that amount.
      - It sets bins that are zero in one distribution (and nonzero in the other) to epsilon.
      - Optionally, it removes bins where the smoothed distribution p is zero.
    
    Parameters
    ----------
    p : array-like
        1D array representing the first probability distribution.
    q : array-like
        1D array representing the second probability distribution.
    is_remove_common_zeros : bool, optional
        If True, remove bins where p is zero after smoothing.
        Default is True.
    
    Returns
    -------
    p : np.ndarray
        The smoothed first distribution.
    q : np.ndarray
        The smoothed second distribution.
    
    Raises
    ------
    AssertionError
        If inputs are not one-dimensional or if there are no nonzero elements.
    """
    # Convert inputs to numpy arrays of float.
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Ensure inputs are one-dimensional.
    assert p.ndim == 1, "Input p must be a one-dimensional array."
    assert q.ndim == 1, "Input q must be a one-dimensional array."
    
    # Create boolean masks for zeros and nonzeros.
    p_zero_mask = (p == 0)
    p_non_zero_mask = ~p_zero_mask
    q_zero_mask = (q == 0)
    q_non_zero_mask = ~q_zero_mask

    # Extended indices: bins where one distribution is zero and the other is nonzero.
    p_extended_idx = p_zero_mask & q_non_zero_mask
    q_extended_idx = q_zero_mask & p_non_zero_mask

    # Extract nonzero elements for each distribution.
    p_non_zero = p[p_non_zero_mask]
    q_non_zero = q[q_non_zero_mask]
    
    # Verify that there are nonzero entries.
    assert p_non_zero.size > 0, "There must be at least one nonzero element in p."
    assert q_non_zero.size > 0, "There must be at least one nonzero element in q."
    
    # Calculate my_eps: 0.001 * min(all nonzero values in p and q).
    combined_nonzero = np.concatenate([p_non_zero.ravel(), q_non_zero.ravel()])
    my_eps = 0.001 * np.min(combined_nonzero)
    
    # Compute the decrease factor for nonzero bins in p.
    p_dec_factor = my_eps * np.sum(p_extended_idx) / float(p_non_zero.size)
    # Assert that subtracting the discount will not produce negative values.
    assert np.all(p[p_non_zero_mask] - p_dec_factor >= 0), "Discount factor too high for p."
    p[p_non_zero_mask] = p[p_non_zero_mask] - p_dec_factor
    p[p_extended_idx] = my_eps
    
    # Compute the decrease factor for nonzero bins in q.
    q_dec_factor = my_eps * np.sum(q_extended_idx) / float(q_non_zero.size)
    assert np.all(q[q_non_zero_mask] - q_dec_factor >= 0), "Discount factor too high for q."
    q[q_non_zero_mask] = q[q_non_zero_mask] - q_dec_factor
    q[q_extended_idx] = my_eps
    
    if is_remove_common_zeros:
        # Remove bins where p is zero (q will be removed correspondingly).
        non_zeros_idx = (p != 0)
        p = p[non_zeros_idx]
        q = q[non_zeros_idx]
    
    # Final assertions: ensure no negative values.
    assert np.all(p >= 0), "Negative values found in p after smoothing."
    assert np.all(q >= 0), "Negative values found in q after smoothing."
    
    return p, q


def kullback_leibler_divergence(p, q, smoothing_method='abs_discount'):
    """
    Compute the Kullback-Leibler divergence between two probability distributions
    with an optional smoothing method.
    
    The KL divergence is computed as:
        KLD(p || q) = sum(p * log(p/q))
    ignoring bins where p is zero (since 0*log(0/q) is defined as zero).
    
    Parameters
    ----------
    p : array-like
        1D array representing the first probability distribution (model distribution).
    q : array-like
        1D array representing the second probability distribution.
    smoothing_method : str, optional
        Smoothing method to apply. Options:
          - 'none': No smoothing; only use bins where p is nonzero.
          - 'abs_discount': Apply absolute discount smoothing (see abs_discount_smoothing).
        Default is 'none'.
    
    Returns
    -------
    float
        The computed Kullback-Leibler divergence.
    
    Raises
    ------
    ValueError
        If an unknown smoothing method is specified.
    AssertionError
        If inputs are not one-dimensional or if there are issues with nonzero elements.
    """
    # Convert inputs to numpy arrays of float.
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Ensure inputs are one-dimensional.
    assert p.ndim == 1, "Input p must be a one-dimensional array."
    assert q.ndim == 1, "Input q must be a one-dimensional array."
    
    if smoothing_method == 'none':
        # Use only bins where p is nonzero.
        mask = (p != 0)
        p = p[mask]
        q = q[mask]
    elif smoothing_method == 'abs_discount':
        # Apply absolute discount smoothing.
        p, q = abs_discount_smoothing(p, q, is_remove_common_zeros=True)
    else:
        raise ValueError("Unknown smoothing method for KLD")
        
    # Ensure that no negative values exist and q has no zeros (for bins where p is positive).
    assert np.all(p >= 0), "p contains negative values after smoothing or masking."
    assert np.all(q > 0), "q must be positive for all bins where p is positive."
    
    # Compute the KL divergence.
    div = np.sum(p * np.log(p / q))
    return div


def symmetric_kullback_leibler_divergence(p, q):
    """
    Compute the symmetric Kullback-Leibler divergence (KLS) between two probability distributions.
    
    Since the standard Kullback-Leibler Divergence (KLD) is not symmetric,
    the symmetric divergence is defined as:
    
        KLS(p, q) = KLD(p || q) + KLD(q || p)
    
    This function uses the existing `kullback_leibler_divergence` function to compute
    the divergence in both directions and then sums them. It accepts parameters for smoothing
    (via absolute discounting) and normalization.
    
    Parameters
    ----------
    p : array-like
        The first distribution as a one-dimensional array of nonnegative numbers.
    q : array-like
        The second distribution as a one-dimensional array of nonnegative numbers.
    discount : float, optional
        The discount value to use in absolute discounting smoothing (default is 0.001).
    apply_smoothing : bool, optional
        If True (default), smoothing is applied to handle zeros in the distributions.
    
    Returns
    -------
    result : float
        - 'kls': the symmetric Kullback-Leibler divergence.
    
    Examples
    --------
    >>> p = [50, 30, 20]
    >>> q = [40, 35, 25]
    >>> result = symmetric_kullback_leibler_divergence(p, q, discount=0.005, apply_smoothing=True)
    """
    
    # Compute KLD(p || q)
    kld_pq = kullback_leibler_divergence(p, q)
    # Compute KLD(q || p)
    kld_qp = kullback_leibler_divergence(q, p)
    
    # Sum both divergences to obtain the symmetric measure
    kls = kld_pq + kld_qp
    
    return kls


def jensen_shannon_divergence(p, q):
    """
    Compute the Jensen-Shannon Divergence (JSD) between two probability distributions.
    
    The JSD is defined as:
    
        JSD(p, q) = 0.5 * KLD(p || m) + 0.5 * KLD(q || m)
    
    where m = (p + q) / 2, and KLD(p || m) is the Kullback-Leibler Divergence from p to m.
    By definition, if p(i) equals 0, the contribution 0 * log(0/m(i)) is taken as 0.
    
    This divergence is always finite and symmetric.
    
    Parameters
    ----------
    p : array-like
        The first distribution as a one-dimensional array of nonnegative numbers.
    q : array-like
        The second distribution as a one-dimensional array of nonnegative numbers.
    
    Returns
    -------
    result : float
        - 'jsd': the computed Jensen-Shannon Divergence.
    
    Raises
    ------
    ValueError
        If the inputs cannot be converted to one-dimensional numpy arrays,
        if they have different lengths, or if they contain negative values.
    
    Examples
    --------
    >>> p = [0.5, 0.3, 0.2]
    >>> q = [0.4, 0.35, 0.25]
    >>> result = jensen_shannon_divergence(p, q)
    """
    # Convert inputs to numpy arrays of floats
    try:
        p_arr = np.asarray(p, dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64)
    except Exception as e:
        raise ValueError("Could not convert inputs to numpy arrays.") from e

    # Check that both arrays are one-dimensional
    if p_arr.ndim != 1 or q_arr.ndim != 1:
        raise ValueError("Both input distributions must be one-dimensional arrays.")
    
    # Check that both arrays have the same length
    if p_arr.size != q_arr.size:
        raise ValueError("Both input distributions must have the same number of elements.")
    
    # Check that there are no negative values
    if np.any(p_arr < 0) or np.any(q_arr < 0):
        raise ValueError("Input distributions must contain nonnegative values only.")
        
    # Compute the average distribution m = (p + q) / 2
    m = 0.5 * (p_arr + q_arr)
    
    # Compute KLD(p || m) using your existing function with smoothing disabled.
    kld_p_m = kullback_leibler_divergence(p_arr, m)
    
    # Compute KLD(q || m)
    kld_q_m = kullback_leibler_divergence(q_arr, m)
    
    # Calculate the Jensen-Shannon Divergence as the average of the two KLDs.
    jsd = 0.5 * kld_p_m + 0.5 * kld_q_m
    
    return jsd

def modified_kolmogorov_smirnov(p, q):
    """
    Compute the Modified Kolmogorov-Smirnov (KS) statistic between two distributions.
    
    The Original KS statistic is defined as:
    
        KSD(p, q) = max_xi | F_p(i) - F_q(i) |
    
    where F_p(i) and F_q(i) are the cumulative distribution functions (CDFs) of p and q.
    
    We define the Modified KS statistic as:
    
        MKS = d = sum_i | p(i) - q(i) |
    
    Parameters
    ----------
    p : array-like
        The first distribution as a one-dimensional array of nonnegative numbers.
    q : array-like
        The second distribution as a one-dimensional array of nonnegative numbers.
    
    Returns
    -------
    result : float
        - 'ksd': the computed Modified Kolmogorov-Smirnov statistic.
    
    Raises
    ------
    ValueError
        If the inputs cannot be converted to one-dimensional numpy arrays,
        if they have different lengths, or if they contain negative values.
    
    Examples
    --------
    >>> p = [0.5, 0.3, 0.2]
    >>> q = [0.4, 0.35, 0.25]
    >>> result = modified_kolmogorov_smirnov(p, q)
    """
    # Convert inputs to numpy arrays of floats
    try:
        p_arr = np.asarray(p, dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64)
    except Exception as e:
        raise ValueError("Could not convert inputs to numpy arrays.") from e

    # Check that both arrays are one-dimensional
    if p_arr.ndim != 1 or q_arr.ndim != 1:
        raise ValueError("Both input distributions must be one-dimensional arrays.")
    
    # Check that both arrays have the same length
    if p_arr.size != q_arr.size:
        raise ValueError("Both input distributions must have the same number of elements.")
    
    # Check that there are no negative values
    if np.any(p_arr < 0) or np.any(q_arr < 0):
        raise ValueError("Input distributions must contain nonnegative values only.")
    
    # Compute the Modified KS statistic as the sum of absolute differences
    
    mks = np.sum(np.abs(p_arr - q_arr))
    
    return mks

def compute_distances_to_reference(pmf_batch: np.ndarray, ref_pmf: np.ndarray) -> dict:
    """
    Compute distance metrics between a batch of PMFs and a reference PMF, channel-by-channel.

    Parameters:
    -----------
    pmf_batch : np.ndarray
        Shape (n_files, n_channels, num_bins) — batch of PMFs
    ref_pmf : np.ndarray
        Shape (n_channels, num_bins) — reference PMF for each channel

    Returns:
    --------
    dict[str, np.ndarray]
        Each value is an array of shape (n_files, n_channels)
    """
    assert pmf_batch.ndim == 3, "pmf_batch must be 3D: (n_files, n_channels, num_bins)"
    assert ref_pmf.ndim == 2, "ref_pmf must be 2D: (n_channels, num_bins)"
    assert pmf_batch.shape[1:] == ref_pmf.shape, "Channel and bin dimensions must match"

    n_files, n_channels, _ = pmf_batch.shape

    results = {
        "chi_square": np.zeros((n_files, n_channels)),
        "correlation": np.zeros((n_files, n_channels)),
        "hellinger": np.zeros((n_files, n_channels)),
        "intersection": np.zeros((n_files, n_channels)),
        "kl_divergence": np.zeros((n_files, n_channels)),
        "symmetric_kl": np.zeros((n_files, n_channels)),
        "jensen_shannon": np.zeros((n_files, n_channels)),
        "modified_ks": np.zeros((n_files, n_channels)),
    }

    for i in range(n_files):
        for ch in range(n_channels):
            p = pmf_batch[i, ch]
            q = ref_pmf[ch]

            results["chi_square"][i, ch] = chi_square_test(p, q)
            results["correlation"][i, ch] = correlation_distance(p, q)
            results["hellinger"][i, ch] = hellinger_distance(p, q)
            results["intersection"][i, ch] = intersection_distance(p, q)
            results["jensen_shannon"][i, ch] = jensen_shannon_divergence(p, q)
            results["symmetric_kl"][i, ch] = symmetric_kullback_leibler_divergence(p, q)
            results["kl_divergence"][i, ch] = kullback_leibler_divergence(p, q)
            results["modified_ks"][i, ch] = modified_kolmogorov_smirnov(p, q)

    return results




# Example Usage:
if __name__ == '__main__': # ! Note MKS is 2x Intersection Distance, so consider removing it (adds no information)
    # Example histograms (probability mass functions)
    p = [0.01, 0.02, 0.03, 0.04, 0.3, 0.2 , 0.4]
    q = [0.02, 0.03, 0.04, 0.05, 0.3, 0.2 , 0.36]
    
    # Compute each distance measure
    chi_stat = chi_square_test(p, q)
    corr_dist = correlation_distance(p, q)
    hellinger_dist = hellinger_distance(p, q)
    inter_dist = intersection_distance(p, q)
    kld = kullback_leibler_divergence(p, q)
    kls = symmetric_kullback_leibler_divergence(p, q)
    jsd = jensen_shannon_divergence(p, q)
    mks = modified_kolmogorov_smirnov(p, q)
    
    # Print the results
    print(f"PMFs: p = {p}, q = {q}")
    print("Distance Measures Results:")
    
    print(f"Chi-Square Statistic: {chi_stat:.4f}")
    print(f"Correlation Distance: {corr_dist:.4f}")
    print(f"Hellinger Distance: {hellinger_dist:.4f}")
    print(f"Intersection Distance: {inter_dist:.4f}")
    print(f"Kullback-Leibler Divergence: {kld:.4f}")
    print(f"Symmetric Kullback-Leibler Divergence: {kls:.4f}")
    print(f"Jensen-Shannon Divergence: {jsd:.4f}")
    print(f"Modified Kolmogorov-Smirnov Statistic: {mks:.4f}")
    
    

