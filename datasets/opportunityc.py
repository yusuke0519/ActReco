import os, sys, zipfile
import wget
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CONFIG = {}
CONFIG['url'] = 'http://www.opportunity-project.eu/system/files/Challenge/OpportunityChallengeLabeled.zip'
CONFIG['out'] = os.path.expanduser('~/datasets')
CONFIG['column_names'] = 'challenge_column_names.txt'
CONFIG['label_legend'] = 'challenge_label_legend.txt'
out_dir = os.path.join(CONFIG['out'], os.path.basename(CONFIG['url']).replace('.zip', ''))


if not os.path.exists(CONFIG['out']):
    os.mkdir(CONFIG['out'])


def download():
    print("Downloading the dataset from {}".format(CONFIG['url']))
    print("{}".format(out_dir + '.zip'))
    wget.download(CONFIG['url'], out=out_dir + '.zip')
    zf = zipfile.ZipFile(out_dir + '.zip', 'r')
    zf.extractall(path=out_dir)
    zf.close()


if not os.path.exists(out_dir):
    download()