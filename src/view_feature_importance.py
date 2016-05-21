#!/usr/bin/env python
# coding=utf-8
from utils import pkl, data_handler
import numpy as np
import logging.config
import sys

sys.path.insert(0, '..')
from configure import *
logging.config.fileConfig('logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))

if __name__ == "__main__":

    importance=pkl.grab(ROOT+'/data/xgb_feature_importance_1_unit')
    name=np.array(pkl.grab(ROOT+'/data/feature_name'))    
    name=name[np.argsort(importance)[::-1]]
    importance=importance[np.argsort(importance)[::-1]]
    for n, i in zip(name, importance):
        print i, n
