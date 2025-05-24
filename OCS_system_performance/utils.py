import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import time

# function to compute the equal error rate with interpolation1d and brentq
def compute_eer(y, y_score): # y is the ground truth, y_score is the prediction score
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1) # pos_label=1 means that the positive class is 1  

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr,kind='linear')(x), 0., 1.) # brentq is a root finding algorithm
    thresh = interp1d(fpr, thresholds)(eer) # interp1d is a linear interpolation function
    return eer, thresh # eer is the equal error rate, thresh is the threshold at eer


def compute_eer_2(label, pred_score, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred_score, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    #print(eer_1)
    #print(eer_2)
    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer_1,eer_2,eer,eer_threshold