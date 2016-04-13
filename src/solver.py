#!/usr/bin/env python
# coding=utf-8

import sys
import logging
import datetime
from utils import evaluate, data_handler

sys.path.insert(0, '..')
from configure import *

def main(solver, type = type) :
    """
    """
    now_time = datetime.datetime.now()
    now_time = datetime.datetime.strftime(now_time, '%Y%m%d-%H:%M:%S')
    os.system('mkdir ' + ROOT + '/result/' + now_time)
    os.system('mkdir ' + ROOT + '/result/' + now_time + '/model')

    training, testing = data_handler.GetData(type = type)
    columns = training.columns.tolist()
    columns.remove('song_id')
    columns.remove('artist_id')
    columns.remove('label_plays')
    train_x = training.ix[:, columns].values
    train_y = training.label_plays.values
    test_x = testing.ix[:, columns].values
    test_y = testing.label_plays.values
    predict = solver(train_x, train_y, test_x, now_time, test_y = test_y, feature_names = columns)
    score = evaluate.evaluate(predict.astype(int).tolist(), test_y.tolist(), testing.artist_id.values.tolist(), testing.label_day.values.astype(int).tolist())
    logging.info('the final score is %.10f' % score)
    with open(ROOT + '/result/' + now_time + '/parameters.param', 'a') as out :
        out.write('score: %.10f\n' % score)
