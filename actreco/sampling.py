# # -*- coding: utf-8 -*-

import numpy as np
import pandas as pd



def get_segment(x, start, stop):
    if isinstance(x, pd.DataFrame):
        return x.iloc[start:stop]
    if isinstance(x, np.ndarray):
        return x[start:stop, ...]
    
    
def get_timesegment(x, start, stop):
    if isinstance(x, pd.DataFrame):
        return x.loc[start:stop]
    if isinstance(x, np.ndarray):
        raise Exception("time segment mode requires pd.Dataframe, not np.ndarray")
    
    
def sliding(x, l_sample, interval):
    start = 0
    stop = start + l_sample
    next_segment = get_segment(x, start, stop)
    while next_segment.shape[0] == l_sample:
        yield next_segment
        start += interval
        stop += interval
        next_segment = get_segment(x, start, stop)


def clip_segment_between(x, start, l_sample, interval=None, nb_timestep=1):
    for t in start:
        segment = [None] * nb_timestep
        if nb_timestep == 1:
            segment[0] = get_segment(x, t, t+l_sample)
        else:
            for i in range(nb_timestep):
                segment[i] = get_segment(x, t+(interval*i), t+l_sample+(interval*i))
                
        yield np.stack(segment)
            
def clip_time_between(x, start, stop):
    assert len(start) == len(stop), "start and stop must have same length"
    
    for t1, t2 in zip(start, stop):
        yield get_timesegment(x, t1, t2)
        
        
def sampling(x, func, dtype='list', **kwargs):
    if isinstance(func, str):
        func = {
            'sliding': sliding,
            'clips': clip_segment_between,
            'clipt': clip_time_between}.get(func)
        assert func is not None
    X = list(func(x, **kwargs))
    if dtype == 'list':
        return X
    elif (dtype == 'ndarray') | (dtype == 'np'):
        return np.array(X)
    else:
        raise Exception("The dtype {} is not supported".format(dtype))


if __name__ == '__main__':
    from datasets.extend_pandas import Accelerations
    df = Accelerations.from_params('sophia2012', route='route1', subject='goto', term='term1', sensor='undersheet')

    # example of sliding window
    X_list = sampling(df.values, sliding, dtype='list', l_sample=400, interval=200)
    X_array = sampling(df.values, sliding, dtype='np', l_sample=400, interval=200)

    # example of clip_time_between
    start = [0, 2, 4, 6, 8]
    stop = [1, 3, 10, 9, 10.2]
    X = sampling(df, clip_time_between, dtype='list', start=start, stop=stop)


