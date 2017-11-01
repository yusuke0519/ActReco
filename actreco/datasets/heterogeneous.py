# # -*- coding: utf-8 -*-
from __future__ import print_function
from six import iteritems
import itertools

import sys, os, zipfile, glob

import wget
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CONFIG = {}

CONFIG['url'] = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity recognition exp.zip'
CONFIG['out'] = os.path.expanduser('~/.keras/datasets')
out_dir = os.path.join(CONFIG['out'], os.path.basename(CONFIG['url']).replace('.zip', ''))


# ALL_USERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
CLASS_ID = {
    'bike': 0, 'sit': 1, 'stairsdown': 2,
    'stairsup': 3, 'stand': 4, 'walk': 5, 'null': 6}


def memoize(f):
    cache = {}
    def helper(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return helper


def download():
    print("Downloading the dataset from {}".format(CONFIG['url']))
    print("{}".format(out_dir + '.zip'))
    wget.download(CONFIG['url'], out=CONFIG['out'], bar=None)
    zf = zipfile.ZipFile(out_dir + '.zip', 'r')
    zf.extractall(path=os.path.dirname(out_dir))
    zf.close()


@memoize
def load_all_data(sensor):
    return pd.DataFrame.from_csv(os.path.join(out_dir, sensor))  # should be loaded onece


def load_file_of(userID='a', sensor='Phones_accelerometer.csv'):
    all_df = load_all_data(sensor)
    temp = all_df[all_df.User == userID]
    x = temp[['x', 'y', 'z']].values
    x = StandardScaler().fit_transform(x)
    
    y = temp['gt'].values
    y = np.array([CLASS_ID[x] for x in y])
    y = y.reshape((len(y), 1))
    y = np.eye(len(CLASS_ID))[y]

    t = temp['Creation_Time']
    return t, x, y


class Heterogeneous(object):
    __name__ = 'usc'
    def __init__(self, userID='a,b,c'):
        self.rawdata = {}
        self.params = {'userID': userID}
        for param in self._paramiter(self.params):
            self.rawdata[str(param)] = load_file_of(**param)

        self.nb_modal = 3
        self.nb_class = len(CLASS_ID)
    
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


if __name__ == "__main__":
    if not os.path.exists(out_dir):
        download()
    dataset = Heterogeneous()
