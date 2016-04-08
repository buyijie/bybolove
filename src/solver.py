#!/usr/bin/env python
# coding=utf-8

import sys
import logging
import datetime
import utils.evaluate as evaluate 

sys.path.insert(0, '..')
from configure import *

def main(dataset, solver) :
    """
    """
    now_time = datetime.datetime.now()
    now_time = datetime.datetime.strftime(now_time, '%Y%m%d-%H:%M:%S')

    training, testing = dataset.GetData()
    columns = training.columns.tolist()
    columns.remove('user_song')
    columns.remove('artist_id')
    columns.remove('label_plays')
    train_x = training.ix[:, columns].values
    train_y = training.label_plays.values
    test_x = testing.ix[:, columns].values
    test_y = testing.label_plays.values
    predict = solver(train_x, train_y, test_x, now_time)
    score = evaluate.evaluate(predict.tolist(), test_y.tolist(), testing.artist_id.values.tolist(), testing.label_day.values.tolist())
    logging.info('the final score is %.10f' % score)
