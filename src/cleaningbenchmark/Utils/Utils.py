"""
This library includes noise injection
decorators and other utils that you can
use.
"""
import numpy as np
import pandas as pd

#This gives you a decorator that you can use to corrupt data
def inject(model, gt=False):
    def decorator(func):
        def function_helper(*args, **kwargs):
            clean = func(*args, **kwargs)
            if gt:
                return clean
            else:
                return model.apply(clean)[0]

        return function_helper

    return decorator


#collects and prints statistics about the
#dirty data
def collect_statistics(data, model, udf, metric, trials=10):
    trials_data = np.zeros((trials,1))
    for i in range(0, trials):
        res = model.apply(data)
        dirty = res[0]
        clean = res[1]
        trials_data[i,1] = metric(udf(dirty), udf(clean))
    print("T= {} runs: {}".format(trials,np.mean(trials_data)))
    return trials_data


def pd_df_diff(df_from, df_to):

    """
    Note that df_from and df_to must share the same index

    Returns: #1 Changes in DataFrame from clean to noised;

             #2 Boolean Matrix with differences 
             highlighted with 'True', for cells;
             
             #3 Boolean Vector with differences 
             highlighted with 'True', for tuples.
    """
    
    ground_truth_errors = df_from != df_to

    ne_stacked = ground_truth_errors.stack()

    changed = ne_stacked[ne_stacked]

    changed.index.names = ['id', 'col']

    diffs_loc = np.where (df_from != df_to)

    changed_from = df_from.values[diffs_loc]

    changed_to = df_to.values[diffs_loc]

    pd_diffs = pd.DataFrame({'from': changed_from, 'to': changed_to}, 
                            index=changed.index)

    return pd_diffs, ground_truth_errors.values, ground_truth_errors.values.any(axis=1)
