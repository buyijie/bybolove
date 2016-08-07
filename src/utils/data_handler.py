#!/usr/bin/env python
# coding=utf-8

import os
import sys
import logging
import pandas as pd

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from configure import *

def GetData(type = 'unit', gap_month = 1, consecutive_recent = [14, 7, 3]) :
    """
    """
#    path = '_cleaning_' + type + '_' + '_'.join(map(str, consecutive_recent)) + '_' + str(gap_month) + '.csv'
    path = '_' + type + '_' + '_'.join(map(str, consecutive_recent)) + '_' + str(gap_month)+'.csv'
    testing_path = ROOT + '/data/final_data_testing' + path
    validation_path = ROOT + '/data/final_data_validation' + path
    training_path = ROOT + '/data/final_data_training' + path
    if not os.path.exists(testing_path) :
        logging.error(testing_path + ' doesn\'t exists!')
        exit(1)
    if not os.path.exists(training_path) :
        logging.error(training_path + ' doesn\'t exists!')
        exit(1)
    if not os.path.exists(validation_path) :
        logging.error(validation_path+ ' doesn\'t exists!')
        exit(1)
    testing = pd.read_csv(testing_path)
    validation = pd.read_csv(validation_path)
    training = pd.read_csv(training_path)
    return training, validation, testing
