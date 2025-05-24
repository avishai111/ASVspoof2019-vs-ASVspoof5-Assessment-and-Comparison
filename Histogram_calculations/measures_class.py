import numpy as np

class measures_class:

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
        return measures_class.kl_div(p1, p2) + measures_class.kl_div(p2, p1)

    @staticmethod
    def hellinger(p1, p2):
        return np.sqrt(1 - np.sum(np.sqrt(p1 * p2)))

   