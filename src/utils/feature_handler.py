#!/usr/bin/env python
# coding=utf-8

import logging
import pandas as pd
from sklearn import preprocessing
import sys
import numpy as np

def binary_feature (data, feature) :
    """
    convert categorical variable into dummy/binary variable by pandas.get_dummies
    parameters:
        @data: original data
        @feature: the feature need to solve
    return:
        $1: the data after handlering
    """
    logging.info ('convert to binary variable for feature: %s' % feature)
    data = pd.concat ([data , pd.get_dummies (data[feature]).rename (columns = lambda x : str (feature) + "_" + str (x))] , axis = 1)
    data.drop(feature, axis = 1, inplace = True)
    return data

def scale_feature (data, feature) :
    """
    standardization of datasets
    make the data look like standard normally distributed data: Gaussian with zero mean and unit variance.
    parameters:
        @data: original data
        @feature: the feature need to solve
    return:
        $1: the data after handlering
    """
    logging.info ('standardization for feature: %s' % feature)
    scaler = preprocessing.StandardScaler ()
    data[feature + '_scaled'] = scaler.fit_transform (data[feature])
    data.drop(feature, axis = 1, inplace = True)
    return data

def Ratio2Plays(ratio, last_month_plays):
    return (ratio+1.0)*last_month_plays

def Plays2Ratio(plays, last_month_plays):
    return (plays-last_month_plays)*1.0/last_month_plays

def Transform(y, transform_type, last_month_plays=None):
    """
    transform labels
    transform_type 0: no transform, 1: ratiolize predict, 2: loglize predict 
    """
    if transform_type==0:
        return y
    elif transform_type==1:
        assert last_month_plays is not None, "must provide last_month_plays for transform plays to ratio"
        return Plays2Ratio(y, last_month_plays)
    elif transform_type==2:
        return np.log(y)

    logging.info("transform_type {} is not defined".transform_type)
    sys.exit(1)

def Convert2Plays(predict, transform_type, last_month_plays=None):
    """
    convert transformed predict back to plays
    transform_type 0: no transform, 1: ratiolize predict, 2: loglize predict
    """
    if transform_type==0:
        return predict
    elif transform_type==1:
        assert last_month_plays is not None, "must provide last_month_plays for converting ratio to plays"
        return Ratio2Plays(predict, last_month_plays)
    elif transform_type==2:
        return np.exp(predict)

    logging.info("transform_type {} is not defined".transform_type)
    sys.exit(1)
