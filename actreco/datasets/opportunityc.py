# coding: utf-8

from __future__ import print_function
from six import iteritems

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


def load_file_of(which='S1', target_key="Gestures", adl_id="ADL1"):
    # TODO: should return pd.Dataframe, or np.Array?
    valid_column_keys=["Accelerometer RUA^"]  # filter out the columns that do not contain at least one of the keys
    file_name = os.path.join(out_dir, "{which}-{adl_id}.dat".format(which=which, adl_id=adl_id))
    data_df = pd.DataFrame.from_csv(file_name, sep=' ', header=None)
    data_df.columns = [column_names()]

    # filter invalid columns
    valid_columns = [x for x in column_names() for y in valid_column_keys + [target_key] if x.find(y) > 0]
    return data_df[valid_columns].fillna(method='ffill').fillna(method='bfill')


def segmentation(data_df, l_sample, interval, target_key="Gestures"):
    """ Segmentating the pd.Dataframe
    :param data_df:
    :return:
    """
    """
    :param data_df:
    :return:
    """
    # time
    t = data_df.index.values
    t = t.reshape(len(t), 1)
    t = mval_segmentation(
        t, 400, l_sample, method='sliding', interval=interval
    )

    # X
    X = data_df.values[:, :-1]  # :-1 corresponds to sensor data
    X = StandardScaler().fit_transform(X)  # Preprocess
    X = mval_segmentation(
        X, 400, l_sample, method='sliding', interval=interval
    )
    X_shape = X.shape
    X = X.reshape((1, X_shape[0], X_shape[1], X_shape[2])).swapaxes(0, 2)

    # y
    y = data_df.values[:, -1]    # :-1 corresponds to label (activity) data
    y = y.reshape((len(y), 1))
    y = mval_segmentation(
        y, 400, l_sample, method='sliding', interval=interval
    )
    labels = np.ones((X_shape[1], len(label_dict(target_key))+1))

    for k, v in iteritems(label_dict(target_key)):
        labels[:, v] = (y == k).sum(axis=2)
    y = labels.argmax(axis=1)
    return {'t': t, 'X': X, 'y': y}


if __name__ == "__main__":
    if not os.path.exists(out_dir):
        download()
    df = load_file_of()
    data_dict = segmentation(df, l_sample=30, interval=15)
    print(data_dict)
    # print(df)