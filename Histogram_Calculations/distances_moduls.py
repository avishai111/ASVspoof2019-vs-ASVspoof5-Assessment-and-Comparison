import numpy as np

class PmfDist:

    @staticmethod
    def ks2_variant(p1, p2):
        return np.sum(np.abs(p1 - p2))

    @staticmethod
    def kl_div(p1, p2):
        # Apply absolute discounting smoothing
        eps = 1e-10
        p1 = np.maximum(p1, eps)
        p2 = np.maximum(p2, eps)
        return np.sum(p1 * np.log(p1 / p2))

    @staticmethod
    def kl_dist(p1, p2):
        return PmfDist.kl_div(p1, p2) + PmfDist.kl_div(p2, p1)

    @staticmethod
    def js_div(p1, p2):
        m = (p1 + p2) / 2
        return (PmfDist.kl_div(p1, m) + PmfDist.kl_div(p2, m)) / 2

    @staticmethod
    def chi_sqr(p1, p2):
        denom = p1 + p2
        numer = (p1 - p2) ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            elements = np.where(denom != 0, numer / denom, 0.0)
        return np.sum(elements)

    @staticmethod
    def hist_intersection(p1, p2):
        return 1 - np.sum(np.minimum(p1, p2))

    @staticmethod
    def hellinger(p1, p2):
        return np.sqrt(1 - np.sum(np.sqrt(p1 * p2)))

    @staticmethod
    def corr(p1, p2):
        p1_centered = p1 - np.mean(p1)
        p2_centered = p2 - np.mean(p2)
        numerator = np.sum(p1_centered * p2_centered)
        denominator = np.sqrt(np.sum(p1_centered**2) * np.sum(p2_centered**2))
        rho = numerator / denominator if denominator != 0 else 0
        return 1 - rho
