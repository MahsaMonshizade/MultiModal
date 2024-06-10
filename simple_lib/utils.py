import difflib
import math
import os
import re
import torch

import numpy as np
import pandas as pd
# from lifelines import KaplanMeierFitter
# from lifelines.statistics import multivariate_logrank_test
# from lifelines.utils import median_survival_times
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, chi2_contingency
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pair_confusion_matrix, accuracy_score


def metagenomics_normalize(x):
    return torch.log2(2 * x + 0.00001)
    



def normalize(x):
    mean_value = x.mean()
    max_value = x.max()
    min_value = x.min()
    return (x - mean_value) / (max_value - min_value)