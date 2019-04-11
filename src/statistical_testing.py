import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from statsmodels.sandbox.stats.runs import mcnemar
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon


def swap_array(y1, y2, p=0.5):
    """
    Swap randomly chosen indices of arrays y1 and y2

    :param y1: numpy array 1
    :param y2: numpy array 2
    :param p: proportion of entries to swap
    :return: list of swapped arrays
    """
    swap_indices = np.random.rand(len(y1)) > p
    new_y1, new_y2 = y1.copy(), y2.copy()
    new_y1[swap_indices] = y2[swap_indices]
    new_y2[swap_indices] = y1[swap_indices]

    return new_y1, new_y2


def permutation_test(y1, y2, ground_truth, metric=accuracy_score, n=10000):
    """ Compute the p-value of hypothesis that predictions from
    model1 are statistically significantly better than predictions
    from model2 with respect to provided metric

    :param y1: numpy array of predictions from model1
    :param y2: numpy array of predictions from model2
    :param ground_truth: numpy array of ground truth value (required for metric computation)
    :param metric: bivariate function that computes a performance metric (e.g.) accuracy
    :param n: number of random permutations performed, higher gives more accurate estimate
        at the cost of running time
    :returns: List containing difference of metric on y1 and y2, the differences for each of
        the n runs and the p-value
    """

    # Get original accuracy scores
    y1_acc = metric(ground_truth, y1)
    y2_acc = metric(ground_truth, y2)
    metric_diff = y1_acc - y2_acc

    metric_perm_diff_list = np.zeros(n)

    for i in range(n):
        # Swap elements of y1 and y2 with probability of 0.5
        y1_perm, y2_perm = swap_array(y1, y2)

        # Compute new accuracies and their difference
        y1_perm_acc = metric(ground_truth, y1_perm)
        y2_perm_acc = metric(ground_truth, y2_perm)

        acc_diff_perm = y1_perm_acc - y2_perm_acc

        metric_perm_diff_list[i] = acc_diff_perm

    p_value = np.mean(metric_perm_diff_list > metric_diff)
    return p_value


def mcnemar_test(y1, y2, ground_truth):
    p_value = mcnemar(y1 == ground_truth, y2 == ground_truth)[1]

    return p_value


def wilcoxon_test(y1, ground_truth):
    z, p_value = wilcoxon(y1, ground_truth)
    print(z)
    return p_value
