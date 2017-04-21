# coding: utf-8

from __future__ import print_function
from six import iteritems
import itertools

import os, sys, zipfile
import wget
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from actreco.segmentation import mval_segmentation

CONFIG = {}
CONFIG['url'] = 'http://www.opportunity-project.eu/system/files/Challenge/OpportunityChallengeLabeled.zip'
CONFIG['out'] = os.path.expanduser('~/datasets')
CONFIG['column_names'] = 'challenge_column_names.txt'
CONFIG['label_legend'] = 'challenge_label_legend.txt'
out_dir = os.path.join(CONFIG['out'], os.path.basename(CONFIG['url']).replace('.zip', ''))

dataset_params = {
    'userID': ['S1', 'S2', 'S3', 'S4'],
    'target_key': ['Locomotion', 'Gestures']
}


if not os.path.exists(CONFIG['out']):
    os.mkdir(CONFIG['out'])


def column_names():
    """ column names of each data file

    In opportunity-dataset, the information is contained in a file 'challenge_column_names.txt'
    """
    file_name = os.path.join(out_dir, CONFIG['column_names'])
    allLines = open(file_name).read().replace('\r', '').split('\n')
    column_name = np.array(
        [x.split(':')[1].split(';')[0] for x in allLines if x.startswith('Column:')])
    return column_name[1:]


def label_dict(target_key):
    file_name = os.path.join(out_dir, CONFIG['label_legend'])
    allLines = open(file_name).read().replace('\r', '') .split('\n')
    label_name = np.array([int(x.replace('\t', '  ').split('   -   ')[0]) for x in allLines if x.find(target_key) > 0])
    label_ids = range(len(label_name))
    return dict(zip(label_name, label_ids))


def download():
    print("Downloading the dataset from {}".format(CONFIG['url']))
    print("{}".format(out_dir + '.zip'))
    wget.download(CONFIG['url'], out=out_dir + '.zip')
    zf = zipfile.ZipFile(out_dir + '.zip', 'r')
    zf.extractall(path=out_dir)
    zf.close()


def load_file_of(userID='S1', target_key="Gestures", adl_id="ADL1"):
    """

    :param userID: name of target user
    :param target_key: type of activity
    :param adl_id: ID of the set (Drill, ADL1, ADL2, ADL3, ADL4, ADL5, ADL6)
    :return: multivariate time-series of timestamp, sensor values, and activity labels
    """
    # TODO: should return pd.Dataframe, or np.Array?
    valid_column_keys=['Acce', 'Iner']  # filter out the columns that do not contain at least one of the keys
    file_name = os.path.join(out_dir, "{which}-{adl_id}.dat".format(which=userID, adl_id=adl_id))
    data_df = pd.DataFrame.from_csv(file_name, sep=' ', header=None)
    data_df.columns = [column_names()]

    # filter invalid columns
    valid_columns = [x for x in column_names() for y in valid_column_keys + [target_key] if x.find(y) > 0]

    data_df = data_df[valid_columns].fillna(method='ffill').fillna(method='bfill')
    timestamps = data_df.index.values
    timestamps = timestamps.reshape(len(timestamps), 1)
    sensor_values = data_df.values[:, :-1]
    activity_labels = data_df.values[:, -1]

    return timestamps, sensor_values, activity_labels


def segmentation(t, X, y, l_sample, interval, target_key="Gestures", preprocessing=True):
    """ Segmentating the pd.Dataframe

    :param data_df:
    :return: segmented time series
    """
    """
    :param data_df:
    :return:
    """
    # time
    t = mval_segmentation(
        t, l_sample, method='sliding', interval=interval
    )
    t = t.swapaxes(0, 1)

    # X
    if preprocessing:
        X = StandardScaler().fit_transform(X)  # Preprocessing
    X = mval_segmentation(
        X, l_sample, method='sliding', interval=interval
    )
    X_shape = X.shape
    X = X.reshape((1, X_shape[0], X_shape[1], X_shape[2])).swapaxes(0, 2)

    # y
    y = y.reshape((len(y), 1))
    y = mval_segmentation(
        y, l_sample, method='sliding', interval=interval
    )

    labels = np.ones((X_shape[1], len(label_dict(target_key))+1))
    for k, v in iteritems(label_dict(target_key)):
        labels[:, v] = (y == k).sum(axis=2)
    y = labels.argmax(axis=1)
    y = y.reshape(len(y), -1)
    return t, X, y
    # return {'t': t, 'X': X, 'y': y}


class Opportunity(object):
    __name__ = 'opportunity'

    def __init__(self, **kwargs):
        self.rawdata = {}
        self.params = kwargs
        for param in self._paramiter(self.params):
            self.rawdata[str(param)] = load_file_of(**param)

    def convert2MLtype(self, l_sample, interval, dim_ordering='bc01'):
        """ convert the raw time-series to ML data type

        :param l_sample: the length of each segment
        :param interval: the interval between each segment
        :return:
        """

        ts, Xs, ys, userIDs = [], [], [], []
        for param in self._paramiter(self.params):
            timestamps, sensor_values, activity_labels = self.rawdata[str(param)]
            t, X, y = segmentation(timestamps, sensor_values, activity_labels, l_sample, interval)
            ts.append(t)
            Xs.append(X)
            ys.append(y)
            userID_array = np.chararray(y.shape)
            userID_array.fill(param['userID'])
            userIDs.append(userID_array)
        ts = np.concatenate(ts)
        Xs = np.concatenate(Xs)
        ys = np.concatenate(ys)
        userIDs = np.concatenate(userIDs)


        return ts, Xs, ys, userIDs

    def _paramiter(self, params):
        _params = []
        for k, v in iteritems(params):
            _params.append((k, v.split(',')))
        param_names, param_variations = zip(*_params)

        for param_variation in itertools.product(*param_variations):
            yield dict(zip(param_names, param_variation))


if __name__ == "__main__":
    if not os.path.exists(out_dir):
        download()
    timestamps, sensor_values, activity_labels = load_file_of()
    data_dict = segmentation(timestamps, sensor_values, activity_labels, l_sample=30, interval=15)

    dataset = Opportunity(userID='S1,S2,S3,S4', target_key='Gestures', adl_id='ADL1,ADL2,ADL3,ADL4,ADL5,Drill')
    t, X, y, userIDs = dataset.convert2MLtype(l_sample=30, interval=15)
    print(X.shape, y.shape)
