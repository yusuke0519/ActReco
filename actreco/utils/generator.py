# # -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

from actreco.sampling import sampling


def choice_func(train=True, initial_test_state=0):
    i = [0]  # nb call
    def choice():
        i[0] += 1
        if train is True:
            return np.random.choice
        
        r = np.random.RandomState(initial_test_state+i[0])
        return r.choice
    return choice()


def batch_generator(x_list, y_list, batch_size, l_sample, train=True, nb_iter=1, categorical=False, seed=0):
    choice = choice_func(train, initial_test_state=seed)
    
    nb_sample = sum([x.shape[0] for x in x_list])
    idx_set = set(range(nb_sample))
    nb_seen_sample = 0
    for x in x_list:
        nb_seen_sample += x.shape[0]
        idx_set -= set(range(nb_seen_sample-l_sample+1, nb_seen_sample+1))
    valid_idx = list(idx_set)
    
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    
    for i in range(nb_iter):
        start = choice(valid_idx, batch_size)
        X = sampling(x, 'clips', dtype='np', start=start, l_sample=l_sample)[..., np.newaxis].swapaxes(1, 3)
        Y = sampling(y, 'clips', dtype='np', start=start, l_sample=l_sample)
        Y = np.concatenate([Y.sum(axis=1), np.ones((batch_size, 1))], axis=-1).argmax(axis=-1)
        if categorical is not False:
            Y = np.eye(categorical)[Y]
        yield (X, Y)
