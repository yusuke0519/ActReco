# coding: utf-8
import numpy as np


def mval_segmentation(data, n_sample, l_sample, method=None, interval=None, buffsize=None, index_sq=None):
    """ segmentate multivariate sequences
    Input:
        data: multivariate data (n_sample, n_feature) ndarray
        n_sample: the number of required sample
        l_sample: the lenght of required sample
        method: how to decide the sampling point
        interval (optional): it is required to segment with 'sliding' manner
        buffsize (optional): it is required to segment with 'max' manner
        index_sq (optional): it is requreid to segment with 'max' manner
    Output:
        out_sqs: segmented multivariate sequences, (n_feature, n_sample, l_sample) ndarray
    """
    if method is None:
        method = 'random'  # how to sample points
    n_total_sample = data.shape[0]
    n_feature = data.shape[1]

    if method is 'random':
        import random
        sample_start = random.sample(
            np.arange(n_total_sample)[:-l_sample], n_sample)
    elif method is 'max':
        sample_start = max_point_sampling(
            np.arange(n_total_sample), n_sample, l_sample,
            index_sq, buffsize=buffsize)
    elif method is 'sliding':
        sample_start = sliding_sampling(
            np.arange(n_total_sample), n_sample, l_sample, interval=interval)
        n_sample = len(sample_start)
        # print("Create %d sample" % (n_sample))
    else:
        raise Exception()
    out_sqs = np.empty((n_feature, n_sample, l_sample))
    for i, ind in enumerate(sample_start):
        out_sqs[:, i, :] = data[ind:ind+l_sample, :].T
    return out_sqs


def max_point_sampling(whole_choice_pos, n_sample, l_sample, index_sq, buffsize=None):
    if buffsize is None:
        buffsize = l_sample/2
    whole_choice_pos = whole_choice_pos[l_sample/2:-l_sample/2]
    whole_choice_value = index_sq[l_sample/2:-l_sample/2]
    whole_choice = dict(zip(whole_choice_pos, whole_choice_value))
    start_sample = []

    while len(start_sample) < n_sample:
        top = sorted(whole_choice.items(), key=lambda x: x[1], reverse=True)[0][0]
        start_sample.append(top-(l_sample/2))
        whole_choice = del_key_in(top-buffsize, top+buffsize+1, whole_choice)
    return start_sample


def sliding_sampling(whole_choice_pos, n_sample, l_sample, interval=None):
    if interval is None:
        interval = l_sample
    n_total_sample = len(whole_choice_pos)
    start_sample = np.arange(0, n_total_sample-l_sample, interval)

    return start_sample


def del_key_in(start, stop, d):
    for k in np.arange(start, stop):
        if k in d.keys():
            del d[k]
    return d