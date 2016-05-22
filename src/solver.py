#!/usr/bin/env python
# coding=utf-8

import sys
import logging
import datetime
from utils import pkl, evaluate, data_handler, feature_reduction

sys.path.insert(0, '..')
from configure import *

song_removed = None

def ReadSongRetainList(type, gap_month) :
    filepath = ROOT + "/data/which_song_removed_" + type + "_" + str(gap_month)
    global song_removed
    if not os.path.exists(filepath) :
        logging.warning(filepath + ' is not exists!')
        song_removed = []
    else :
        song_removed = []
        file = open(filepath)
        for each in file :
            song_removed.append(each)

def HandlePredict(predict, song_id = None) :
    """
    """
    # FLOOR value
    # negative -> zero
# max(1, int(v))????
    predict = map(lambda v : max(0, int(v)), predict)
    if song_id is None : return predict
    assert len(song_id) == len(predict)
    global song_removed
    for i in xrange(len(song_id)) :
        if song_id[i] in song_removed:
            predict[i] = 0
    return predict

def main(solver, filepath,  gap_month = 1, type = 'unit', dimreduce_func = feature_reduction.undo, since_when=201503, transform_type=0) :
    """
    """
    ReadSongRetainList(type, gap_month)


    os.system('mkdir ' + filepath)
    os.system('mkdir ' + filepath + '/model')

    training, validation, testing = data_handler.GetData(gap_month = gap_month, type = type)

#Add numerical artist_id and song_id as feature
    training['artist_id_category']=training['artist_id'].astype('category')
    training['artist_id_numeric']=training['artist_id_category'].cat.codes
    training.drop('artist_id_category', axis=1, inplace=True)
    training['song_id_category']=training['song_id'].astype('category')
    training['song_id_numeric']=training['song_id_category'].cat.codes
    training.drop('song_id_category', axis=1, inplace=True)

    validation['artist_id_category']=validation['artist_id'].astype('category')
    validation['artist_id_numeric']=validation['artist_id_category'].cat.codes
    validation.drop('artist_id_category', axis=1, inplace=True)
    validation['song_id_category']=validation['song_id'].astype('category')
    validation['song_id_numeric']=validation['song_id_category'].cat.codes
    validation.drop('song_id_category', axis=1, inplace=True)

    testing['artist_id_category']=testing['artist_id'].astype('category')
    testing['artist_id_numeric']=testing['artist_id_category'].cat.codes
    testing.drop('artist_id_category', axis=1, inplace=True)
    testing['song_id_category']=testing['song_id'].astype('category')
    testing['song_id_numeric']=testing['song_id_category'].cat.codes
    testing.drop('song_id_category', axis=1, inplace=True)

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

    columns.remove('artist_id_numeric')
    columns.remove('song_id_numeric')

#delete some features that i think not important
    """
    del_name=[]
    for _name in columns:
        if ('Morning' in _name) or ('Noon' in _name) or ('Afternoon' in _name) or ('Evening' in _name) or ('Midnight' in _name):
            del_name.append(_name)

    for _name in del_name:
        columns.remove(_name)
    """

#    pkl.store(columns, ROOT+'/data/feature_name')
#use data after month since_when to train model
    rows_train=training.month.values>since_when

    train_x = training.ix[rows_train, columns].values
    train_y = training.ix[rows_train, :].label_plays.values
    validation_x = validation.ix[:,columns].values
    validation_y = validation.label_plays.values
    test_x = testing.ix[:, columns].values
    # feature reduction
    train_x, validation_x, test_x, columns = dimreduce_func(train_x, train_y, validation_x, validation_y, test_x, columns, gap_month = gap_month, type = type)
    predict_validation, predict_test = solver(train_x, train_y, validation_x, test_x, filepath, validation_y = validation_y, feature_names = columns, validation_artist_id=validation.artist_id.values.tolist(), validation_month=validation.month.values.astype(int).tolist(), validation_label_day=validation.label_day.values.astype(int).tolist(), transform_type=transform_type, validation_song_id=validation.song_id.values)
    predict_validation = HandlePredict(predict_validation.tolist(), validation.song_id.values)
    predict_test = HandlePredict(predict_test.tolist(), testing.song_id.values)
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
    main(solver, filepath=filepath + "/1", gap_month=1, type=type, dimreduce_func = feature_reduction.undo, transform_type=0) 
    main(solver, filepath=filepath + "/2", gap_month=2, type=type, dimreduce_func = feature_reduction.undo, transform_type=0)
    evaluate.mergeoutput(filepath)
