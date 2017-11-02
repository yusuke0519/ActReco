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


def batch_generator(x_list, y_list, batch_size, l_sample, interval=None, nb_timestep=1, train=True, nb_iter=1, categorical=False, seed=0):
    choice = choice_func(train, initial_test_state=seed)
    
    nb_sample = sum([x.shape[0] for x in x_list])
    idx_set = set(range(nb_sample))
    nb_seen_sample = 0
    for x in x_list:
        nb_seen_sample += x.shape[0]
        if nb_timestep == 1:
            idx_set -= set(range(nb_seen_sample-l_sample+1, nb_seen_sample+1))
        else:
            idx_set -= set(range(nb_seen_sample-(l_sample+1+(interval)*(nb_timestep-1)), nb_seen_sample+1))
    valid_idx = list(idx_set)
    
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    
    for i in range(nb_iter):
        start = choice(valid_idx, batch_size)
        X = sampling(x, 'clips', dtype='np', start=start, l_sample=l_sample, interval=interval, nb_timestep=nb_timestep)[..., np.newaxis].swapaxes(2, 4)
        Y = sampling(y, 'clips', dtype='np', start=start, l_sample=l_sample, interval=interval, nb_timestep=nb_timestep)
        Y = np.concatenate([Y.sum(axis=2), np.ones((batch_size, nb_timestep, 1))], axis=-1).argmax(axis=-1)
        if categorical is not False:
            Y = np.eye(categorical)[Y]

        if nb_timestep == 1:
            X, Y = X.squeeze(1), Y.squeeze(1) 
        yield (X, Y)
