# # -*- coding: utf-8 -*-

from __future__ import print_function
from six import iteritems
import itertools

import sys, os, zipfile, glob

import wget
import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler

from timeseries.segmentation import mval_segmentation

CONFIG = {}
CONFIG['url'] = 'http://sipi.usc.edu/HAD/USC-HAD.zip'
CONFIG['out'] = os.path.expanduser('~/.keras/datasets')
out_dir = os.path.join(CONFIG['out'], os.path.basename(CONFIG['url']).replace('.zip', ''))


def load_file_of(userID='1'):
    file_list = glob.glob(os.path.join(out_dir, "Subject{}".format(userID), "a*.mat"))
    sensor_values = [None] * len(file_list)
    activity_labels = [None] * len(file_list)
    timestamps = [None] * len(file_list)
    for i, f in enumerate(file_list):
        data = scipy.io.loadmat(f)

        sv = data['sensor_readings']
        sv = StandardScaler().fit_transform(sv)
        al = [int(data['activity_number'][0])] * len(sv)
        t = range(len(sv))
        sensor_values[i] = sv
        activity_labels[i] = al
        timestamps[i] = t

    
    sensor_values = np.concatenate(sensor_values, axis=0)
    activity_labels = np.concatenate(activity_labels).astype('int')
    timestamps = np.concatenate(timestamps)
    return timestamps, sensor_values, activity_labels


class USC(object):
    __name__ = 'opportunity'

    def __init__(self, userID='1'):
        self.rawdata = {}
        self.params = {
            "userID": userID,
        }
        for param in self._paramiter(self.params):
            self.rawdata[str(param)] = load_file_of(**param)

        self.nb_modal = 6
        self.nb_class = 12

    def _paramiter(self, params):
        _params = []
        for k, v in iteritems(params):
            _params.append((k, v.split(',')))
        param_names, param_variations = zip(*_params)

        for param_variation in itertools.product(*param_variations):
            yield dict(zip(param_names, param_variation))
    
    def get(self, name):
        idx = {'T': 0, 't': 0, 'X': 1, 'x': 1, 'Y': 2, 'y': 2}.get(name)
        assert idx is not None, "argument name must be {'T', 't', 'X', 'x', 'Y', 'y'}. Given {}".format(name)

        return [x[idx] for x in list(self.rawdata.values())]


if __name__ == '__main__':
    dataset = USC()
