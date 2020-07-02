import datatable as dt
import pandas as pd
import altair as alt
import numpy as np
import pandas_bokeh as pb

from scipy import mean
from datatable import f,by
from itertools import repeat
from sklearn.utils import shuffle,resample
from scipy.stats import sem, t

import random

pd.set_option('plotting.backend', 'pandas_bokeh')
pb.output_notebook()


def _infer_specify(DT,response):
    return DT[:,response]

def _infer_hypothesis(DT,null_type,**kwargs):
    res_var = DT.names[0]
    null_point = [ point_val for point_val in kwargs.values()]
    obs_mean = np.mean(DT[:,f[res_var]].to_list()[0])
    shift = null_point - obs_mean
    
    if null_type=="point":
        DT[:,f[res_var]]=DT[:,f[res_var]+shift]
        return DT

def _infer_generate(DT,reps,sample_type):
    if sample_type=="bootstrap":
        rep_size = len(resample(DT.to_list()[0]))
        res_var = DT.names[0]
        rep_dict={'reps':[],res_var:[]}
        for rep in range(0,reps):
            rep_dict['reps'].extend([ v for v in repeat(rep,rep_size)])
            rep_dict[res_var].extend(resample(DT.to_list()[0],replace=True,n_samples=rep_size))
        return dt.Frame(rep_dict)
    elif sample_type=="permute":
        print(f'The given method {sample_type} is not yet implemented')
    else:
        pass
    
def _infer_caluclate(DT,stat):
    if stat == 'mean':
        return DT[:,{'mean_val':dt.mean(f[1])},by(f[0])]
    elif stat == 'median':
        return DT[:,{'median_val':dt.median(f[1])},by(f[0])]
    else:
        pass
    
def _infer_visualize(DT):
    
    df= DT[:,f[1]].to_pandas()
    cols = df.shape[0]
    min_v = int(round(df.iloc[:,0].min()))
    max_v = int(round(df.iloc[:,0].max()))
    p = df.plot_bokeh.hist(title="Null Distribution",fill_color="green",vertical_xlabel=True)
    
    return p

def _infer_confidence_intervals(DT,conf_level=0.95):
    data = DT[:,f[1]].to_list()[0]
    n=len(data)
    point_est = np.mean(data)
    std_err= sem(data)
    h= std_err * t.ppf((1+conf_level)/2,n-1)
    return [ round(point_est - h,3) , round(point_est + h,3) ]


def _define_variables(variables):
    # Check that variable names are passed in a list.
    # Can take None as value
    if not variables or isinstance(variables, list):
        variables = variables
    else:
        variables = [variables]
    return variables