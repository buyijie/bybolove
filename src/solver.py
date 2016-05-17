#!/usr/bin/env python
# coding=utf-8

import sys
import logging
import datetime
from utils import evaluate, data_handler, feature_reduction

sys.path.insert(0, '..')
from configure import *

def HandlePredict(predict) :
    """
    """
    # FLOOR value
    # negative -> zero
# max(1, int(v))????
    predict = map(lambda v : max(0, int(v)), predict)
    return predict

def main(solver, filepath,  gap_month = 1, type = 'unit', dimreduce_func = feature_reduction.undo, since_when=201503, transform_type=0) :
    """
    """
    os.system('mkdir ' + filepath)
    os.system('mkdir ' + filepath + '/model')

    training, validation, testing = data_handler.GetData(gap_month = gap_month, type = type)
    training.last_month_plays += 1
    validation.last_month_plays += 1
    testing.last_month_plays += 1
    training.label_plays += 1
    validation.label_plays += 1
    columns = training.columns.tolist()
    columns.remove('month')
    columns.remove('song_id')
    columns.remove('artist_id')
    columns.remove('label_plays')
#use data after month since_when to train model
    rows_train=training.month.values>since_when

    train_x = training.ix[rows_train, columns].values
    train_y = training.ix[rows_train, :].label_plays.values
    validation_x = validation.ix[:,columns].values
    validation_y = validation.label_plays.values
    test_x = testing.ix[:, columns].values
    # feature reduction
    train_x, validation_x, test_x, columns = dimreduce_func(train_x, train_y, validation_x, validation_y, test_x, columns, gap_month = gap_month, type = type)
    predict_validation, predict_test = solver(train_x, train_y, validation_x, test_x, filepath, validation_y = validation_y, feature_names = columns, validation_artist_id=validation.artist_id.values.tolist(), validation_month=validation.month.values.astype(int).tolist(), validation_label_day=validation.label_day.values.astype(int).tolist(), transform_type=transform_type)
    predict_validation = HandlePredict(predict_validation.tolist())
    predict_test = HandlePredict(predict_test.tolist())
    score = evaluate.evaluate(predict_validation, validation_y.tolist(), validation.artist_id.values.tolist(), validation.month.values.astype(int).tolist(), validation.label_day.values.astype(int).tolist())
    evaluate.output(ROOT + '/predict_' + str(gap_month) ,predict_test, testing.artist_id.values.tolist(), testing.month.values.astype(int).tolist(), testing.label_day.values.astype(int).tolist())
    logging.info('the final score is %.10f' % score)
    with open(filepath + '/parameters.param', 'a') as out :
        out.write('score: %.10f\n' % score)

def run(solver, type = 'unit') :
    """
    """
    now_time = datetime.datetime.now()
    now_time = datetime.datetime.strftime(now_time, '%Y%m%d-%H%M%S')
    filepath = ROOT + '/result/' + now_time
    os.system('mkdir ' + filepath)
    main(solver, filepath=filepath + "/1",  gap_month=1, type=type, dimreduce_func = feature_reduction.undo, transform_type=0) 
    main(solver, filepath=filepath + "/2", gap_month=2, type=type, dimreduce_func = feature_reduction.undo, transform_type=0)
    evaluate.mergeoutput(filepath)
