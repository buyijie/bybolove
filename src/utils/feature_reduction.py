#!/usr/bin/env python
# coding=utf-8

import sys
import logging
import pandas as pd
import numpy as np
import utils.pkl as pkl
from sklearn.decomposition import PCA 
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from configure import *

def pca_solver (data, K = PCACOMPONENT) :
    """
    Linear dimensionality reduction by pricipal component analysis
    @parameters:
        data: original data (dataFrame)
    @return:
        $1: the data after handlering (ndarray)
    """
    logging.info ('begin to run pca')
    logging.info ('the number of components in pca is %d' % K)
    pca = PCA (n_components = K , whiten = True)
    if type (data) is pd.DataFrame :
        data_values = data.values
    else :
        data_values = data
    pca.fit (data_values)
    pca_data = pca.transform (data_values)
    logging.info ('finished pca')
    return pca_data

def pca (train_x,  train_y, validation_x, validation_y, test_x, feature_name, gap_month = 1, type = 'unit') :
    """
    """
    if train_x.shape[0] != train_y.shape[0] or validation_x.shape[0] != validation_y.shape[0] :
        logging.error('the size of data set is mismatch')
        exit(-1)
    if train_x.shape[1] != validation_x.shape[1] :
        logging.error('the number of feature in different data set is mismatch')
        exit(-1)
    
    pca_data = np.vstack ([train_x, validation_x, test_x])
    pca_data = pca_solver (pca_data)
    new_feature_name = [str(i + 1) for i in xrange(pca_data.shape[1])]
    logging.info('finished feature reduction, the original feature is :')
    print feature_name
    logging.info('the new feature is :')
    print new_feature_name
    return pca_data[:train_x.shape[0],:], pca_data[train_x.shape[0]:-test_x.shape[0]], pca_data[-test_x.shape[0]:], new_feature_name

def gbdt_feature_importance (train, label, gap_month = 1, type = 'unit') :
    filepath = ROOT + '/data/feature_importance_%d_%s' % (gap_month, type)
    if os.path.exists (filepath) :
        logging.info (filepath + ' exists!')
        feature_importance = pkl.grab (filepath)
    else :
        logging.info ('feature_importance start!')
        logging.info ('the size of data used to cal feature importance is (%d %d)' % train.shape)
        gb = GradientBoostingRegressor(n_estimators = 500 , learning_rate = 0.03 , max_depth = 3 , random_state = 1000000007, verbose = 1).fit (train, label)
        feature_importance = gb.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max ())
        pkl.store (feature_importance, filepath)
    return feature_importance

def gbdt_dimreduce_threshold (train_x, train_y, validation_x, validation_y, test_x,  feature_name, feature_threshold = GBDTFEATURETHRESHOLD, gap_month = 1, type = 'unit') :
    """
    """
    logging.info ('begin gbdt_dimreduce_threshold')
    logging.info ('before gbdt dim-reducing : (%d %d)' % (train_x.shape))
    data = np.vstack([train_x, validation_x])
    label = np.hstack([train_y, validation_y])
    feature_importance = gbdt_feature_importance (data, label, gap_month = gap_month, type = type)
    important_index = np.where (feature_importance > feature_threshold)[0]
    sorted_index = np.argsort (feature_importance[important_index])[::-1]

    new_train = train_x[:,important_index][:,sorted_index]
    new_validation = validation_x[:,important_index][:,sorted_index]
    new_test = test_x[:,important_index][:,sorted_index]
    new_feature_name = [feature_name[i] for i in important_index]
    new_feature_name = [new_feature_name[i] for i in sorted_index]
    logging.info ('after gbdt dim-reducing : (%d %d)' % (new_train.shape))
    logging.info('finished feature reduction, the original feature is :')
    print feature_name
    logging.info('the new feature is :')
    print new_feature_name
    return new_train, new_validation, new_test, new_feature_name

def gbdt_dimreduce_number (train_x, train_y, validation_x, validation_y, test_x, feature_name, feature_number = GBDTFEATURENUMBER, gap_month = 1, type = 'unit') :
    """
    """
    logging.info ('begin gbdt_dimreduce_number')
    logging.info ('before gbdt dim-reducing : (%d %d)' % (train_x.shape))
    data = np.vstack([train_x, validation_x])
    label = np.hstack([train_y, validation_y])
    feature_importance = gbdt_feature_importance (data, label, gap_month = gap_month, type = type)
    sorted_index = np.argsort (feature_importance)[::-1]
    sorted_index = sorted_index[:feature_number]
    
    new_train = train_x[:,sorted_index]
    new_validation = validation_x[:,sorted_index]
    new_test = test_x[:,sorted_index]
    new_feature_name = [feature_name[i] for i in sorted_index]
    logging.info ('after gbdt dim-reducing : (%d %d)' % (new_train.shape))
    logging.info('finished feature reduction, the original feature is :')
    print feature_name
    logging.info('the new feature is :')
    print new_feature_name
    return new_train, new_validation, new_test, new_feature_name

def undo (train_x, train_y, validation_x, validation_y, test_x, feature_name, gap_month = 1, type = 'unit') :
    """
    nothing to do
    """
    logging.info('no feature reduction')
    return train_x, validation_x, test_x, feature_name

if __name__ == '__main__' :
    pass
