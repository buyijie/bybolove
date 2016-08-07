#!/usr/bin/env python
# coding=utf-8

import sys
import logging
import datetime
from utils import pkl, evaluate, data_handler, feature_reduction, feature_handler
import numpy as np
import pandas as pd
import math

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
            song_removed.append(each.strip())

def HandlePredict(predict, song_id = None) :
    """
    """
    # FLOOR value
    # negative -> zero
# max(1, int(v))????
    song_id=None
    predict = map(lambda v : max(0, int(v)), predict)
    if song_id is None : return predict
    assert len(song_id) == len(predict)
    global song_removed
    for i in xrange(len(song_id)) :
        if song_id[i] in song_removed:
            predict[i] = 0
    return predict

def main(solver, filepath,  gap_month = 1, type = 'unit', artist_selected=np.array([]), dimreduce_func = feature_reduction.undo, since_when=201501, transform_type=0) :
    """
    """

    os.system('mkdir ' + filepath)
    os.system('mkdir ' + filepath + '/model')

    training, validation, testing = data_handler.GetData(gap_month = gap_month, type = type)

    training=training[training.artist_id.isin(artist_selected)]
    validation=validation[validation.artist_id.isin(artist_selected)]
    testing=testing[testing.artist_id.isin(artist_selected)]
    
#select Covariance of only P2 artists, comment out below lines if not needed
#    del_name_0610=[]
#    artist_list_0610=np.array(pd.read_csv(ROOT+'/result/artist_list.csv').artist_id.tolist())
#    artist_list_2_0610=np.array(pd.read_csv(ROOT+'/result/artist_list_P2.csv').artist_id.tolist())
#    for _artist_id in xrange(100):
#        if artist_list_0610[_artist_id] not in artist_list_2_0610:
#            del_name_0610.append('CovarianceBetweenSongAndArtist_'+str(_artist_id)+'_mul_total_plays_for_every_artist_all_'+str(_artist_id))
#
#    for _name in del_name_0610:
#        training.drop(_name, axis=1, inplace=True)
#        validation.drop(_name, axis=1, inplace=True)
#        testing.drop(_name, axis=1, inplace=True)

   
#random select 20% data as validation, comment below code if u think not necessary
    _tmp=np.vstack([training.values, validation.values])
    _indices=np.random.permutation(_tmp.shape[0])   
    _tmp=_tmp[_indices, :]
    _split=int(_tmp.shape[0]*0.7)
    _train=_tmp[:_split, :]
    _validation=_tmp[_split:, :]
    training=pd.DataFrame(_train)
    training.columns=testing.columns.tolist()
    validation=pd.DataFrame(_validation)
    validation.columns=testing.columns.tolist()

#----------------get dummies start-------------------------------------------------
    print training.shape
    print validation.shape
    print testing.shape

    training['label_week']=training['label_day']//7
    validation['label_week']=validation['label_day']//7
    testing['label_week']=testing['label_day']//7

    _dummy_list=[
        'artist_id', 
        'label_week', 'Gender', 'Language', 'label_weekday']

    for _feature in _dummy_list:
        _all=pd.concat([training[_feature], validation[_feature], testing[_feature]], ignore_index=True, axis=0)
        _all=pd.DataFrame(_all)
        _all[_feature]=_all[_feature].astype('category')
        _all=feature_handler.binary_feature(_all, _feature)
        _feature_names=_all.columns.tolist()
        _all_values=_all.values
        for _feature_index in xrange(_all_values.shape[1]):
            training[_feature_names[_feature_index]]=_all_values[:training.shape[0], _feature_index]
            validation[_feature_names[_feature_index]]=_all_values[training.shape[0]: training.shape[0]+validation.shape[0], _feature_index]
            testing[_feature_names[_feature_index]]=_all_values[training.shape[0]+validation.shape[0]:, _feature_index]

    print training.shape
    print validation.shape
    print testing.shape

#---------------get dummies end---------------------------------------------------

#---------------handle/scale some numerical features start------------------------ 

    training['published_days']=training['published_days'].map(lambda v: math.exp(-1.*v/10.))
    validation['published_days']=validation['published_days'].map(lambda v: math.exp(-1.*v/10.))
    testing['published_days']=testing['published_days'].map(lambda v: math.exp(-1.*v/10.))

    _month_days={201503: 31., 201504: 30., 201505: 31., 201506: 30., 201507: 31., 201508: 30.}
    training['month_days']=training['month'].map(lambda v: _month_days[v-gap_month])
    validation['month_days']=validation['month'].map(lambda v: _month_days[v-gap_month])
    testing['month_days']=testing['month'].map(lambda v: _month_days[v-gap_month])
    
    training['total_plays_for_one_song_all']=training['total_plays_for_one_song_all']/training['month_days']
    validation['total_plays_for_one_song_all']=validation['total_plays_for_one_song_all']/validation['month_days']
    testing['total_plays_for_one_song_all']=testing['total_plays_for_one_song_all']/testing['month_days']

    training['total_plays_for_artist_all']=training['total_plays_for_artist_all']/training['month_days']
    validation['total_plays_for_artist_all']=validation['total_plays_for_artist_all']/validation['month_days']
    testing['total_plays_for_artist_all']=testing['total_plays_for_artist_all']/testing['month_days']

#todo, parameters to modified
#    for _artist_id in xrange(100):
#        training['CovarianceBetweenSongAndArtist_'+str(_artist_id)+'_mul_total_plays_for_every_artist_all_'+str(_artist_id)]=training['CovarianceBetweenSongAndArtist_'+str(_artist_id)+'_mul_total_plays_for_every_artist_all_'+str(_artist_id)]/training['month_days']
#        validation['CovarianceBetweenSongAndArtist_'+str(_artist_id)+'_mul_total_plays_for_every_artist_all_'+str(_artist_id)]=validation['CovarianceBetweenSongAndArtist_'+str(_artist_id)+'_mul_total_plays_for_every_artist_all_'+str(_artist_id)]/validation['month_days']
#        testing['CovarianceBetweenSongAndArtist_'+str(_artist_id)+'_mul_total_plays_for_every_artist_all_'+str(_artist_id)]=testing['CovarianceBetweenSongAndArtist_'+str(_artist_id)+'_mul_total_plays_for_every_artist_all_'+str(_artist_id)]/testing['month_days']

    for _days in [14, 7, 3]:
        training['total_plays_for_one_song_recent_'+str(_days)]=training['total_plays_for_one_song_recent_'+str(_days)]*1.0/_days
        validation['total_plays_for_one_song_recent_'+str(_days)]=validation['total_plays_for_one_song_recent_'+str(_days)]*1.0/_days
        testing['total_plays_for_one_song_recent_'+str(_days)]=testing['total_plays_for_one_song_recent_'+str(_days)]*1.0/_days
    
#---------------handle/scale some numerical features end--------------------------

    print training.shape
    print validation.shape
    print testing.shape

    assert (training.shape[1]==validation.shape[1]) and (training.shape[1]==testing.shape[1]), 'feature numbers shoule be equal'

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
    columns.remove('label_day')
    columns.remove('month_days')
    columns.remove('Gender')
    columns.remove('Language')
    columns.remove('label_weekday')
    columns.remove('label_week')
#remove features for month 2
    if (gap_month==2): 
        remove_list_for_month_2=[]
        for _name in columns:
            if ('Covariance' in _name) or ('div' in _name):
                remove_list_for_month_2.append(_name)
        for _name in remove_list_for_month_2:
            columns.remove(_name)

#delete some features that i think not important
    del_name=[]
    for _name in columns:
        if ('Morning' in _name) or ('Noon' in _name) or ('Afternoon' in _name) or ('Evening' in _name) or ('Midnight' in _name):
            del_name.append(_name)

    for _name in del_name:
        columns.remove(_name)

#    pkl.store(columns, ROOT+'/data/feature_name')
#use data after month since_when to train model, !!!!deprecated!!!!!!
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
#comment below four lines of code, when random 20% validation set
#    score, bad_predict_artists = evaluate.evaluate(predict_validation, validation_y.tolist(), validation.artist_id.values.tolist(), validation.month.values.astype(int).tolist(), validation.label_day.values.astype(int).tolist())
#    logging.info('the final score is %.10f' % score)
#    with open(filepath + '/parameters.param', 'a') as out :
#        out.write('score: %.10f\n' % score)
    evaluate.output(ROOT + '/predict_' + str(gap_month) ,predict_test, testing.artist_id.values.tolist(), testing.month.values.astype(int).tolist(), testing.label_day.values.astype(int).tolist())

#    print 'bad_predict_artists in gap {} is :'.format(gap_month)
#    with open(filepath + '/bad_predict_artists.log', 'w') as out:
#        for artist in bad_predict_artists:
#            out.write(artist+'\n')
#            print artist
#    evaluate.plot_artist_daily_error(predict_validation, validation_y.tolist(), validation.artist_id.values.tolist(), validation.month.values.astype(int).tolist(), validation.label_day.values.astype(int).tolist(), filepath)
#    evaluate.plot_all_daily_error(predict_validation, validation_y.tolist(), validation.artist_id.values.tolist(), validation.month.values.astype(int).tolist(), validation.label_day.values.astype(int).tolist(), filepath)


def run(solver, type = 'unit') :
    """
    """
    now_time = datetime.datetime.now()
    now_time = datetime.datetime.strftime(now_time, '%Y%m%d-%H%M%S')
    filepath = ROOT + '/result/' + now_time
    os.system('mkdir ' + filepath)
#todo parameters to modified
    _selected_artist_number=45
    _model_number=50
    for i in xrange(_model_number):
        _filepath=filepath+'/run_'+str(i)
        os.system('mkdir '+_filepath)
        artist_list=pd.read_csv(ROOT+'/result/artist_list_P2.csv')
        artist_all=np.array(artist_list.artist_id.tolist())
        order_artist=np.random.permutation(len(artist_list))
        artist_selected=artist_all[order_artist][:_selected_artist_number]
        logging.info(artist_selected)
        main(solver, filepath=_filepath + "/1", gap_month=1, type=type, artist_selected=artist_selected, dimreduce_func = feature_reduction.undo, transform_type=0) 
#        main(solver, filepath=_filepath + "/2", gap_month=2, type=type, artist_selected=artist_selected, dimreduce_func = feature_reduction.undo, transform_type=0)
        evaluate.mergeoutput(_filepath)
