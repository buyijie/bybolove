#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
import logging
import datetime
from feature_handler import *

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from configure import *

def DateToStr(date) :
    """
    """
    return date.strftime('%Y%m%d')

class DataSet :
    """
    """
    def __init__ (self, type = "unit") :
        self.type_ = type
        self.Read()
        self.Join()
        self.GetArtistSet()
        self.GetTimePeriod()
        self.GetDataStatistics()

    def Read(self) :
        """
        read data from csv file
        """
        logging.info('read data from csv file, the type is %s'%self.type_)
        if self.type_ == "unit" :
            self.songs_ = pd.read_csv(ROOT + '/data/mars_tianchi_songs_tiny.csv', header = None)
            self.action_train_ = pd.read_csv(ROOT + '/data/mars_tianchi_user_actions_tiny_training.csv', header = None)
            self.action_test_ = pd.read_csv(ROOT + '/data/mars_tianchi_user_actions_tiny_testing.csv', header = None)
            self.label_ = pd.read_csv(ROOT + '/data/label_tiny.csv', header = None)
        elif self.type_ == 'full' :
            self.songs_ = pd.read_csv(ROOT + '/data/mars_tianchi_songs.csv', header = None)
            self.action_train_ = pd.read_csv(ROOT + '/data/mars_tianchi_user_actions_training.csv', header = None)
            self.action_test_ = pd.read_csv(ROOT + '/data/mars_tianchi_user_actions_testing.csv', header = None)
            self.label_ = pd.read_csv(ROOT + '/data/label.csv', header = None)
        else:
            logging.error('Invalid type of data set, please choose unit or full')
            exit (1)

        self.songs_.columns = ['song_id', 'artist_id', 'publish_time', 'song_init_plays', 'Language', 'Gender']
        self.action_train_.columns = ['user_id', 'song_id', 'gmt_create', 'action_type', 'Ds']
        self.action_test_.columns = ['user_id', 'song_id', 'gmt_create', 'action_type', 'Ds']
        self.label_.columns = ['artist_id', 'Plays', 'Ds']

    def Join(self) :
        """
        """
        logging.info("merge action and songs by song_id")
        self.train_ = pd.merge(self.action_train_, self.songs_, on = 'song_id')
        self.test_ = pd.merge(self.action_test_, self.songs_, on = 'song_id')

    def GetArtistSet(self) :
        """
        """
        self.artist_set_ = set(self.train_.artist_id.values)
        logging.info('there are %d artists'%len(self.artist_set_))

    def GetTimePeriod(self) :
        """
        get the time period in training set and validation set
        """
        logging.info('get time period')
        time_period = sorted(map(lambda v: datetime.datetime.strptime(str(v), '%Y%m%d'), set(self.train_.Ds.values)))
        self.train_time_period_ = [time_period[0], time_period[-1]] 
        logging.debug('the time period in training set: %s to %s'%(time_period[0].strftime("%Y-%m-%d"), time_period[-1].strftime("%Y-%m-%d")))
        time_period = sorted(map(lambda v: datetime.datetime.strptime(str(v), '%Y%m%d'), set(self.test_.Ds.values)))
        self.test_time_period_ = [time_period[0], time_period[-1]]
        logging.debug('the time period in testing set: %s to %s'%(time_period[0].strftime("%Y-%m-%d"), time_period[-1].strftime("%Y-%m-%d")))

    def GetDataStatistics(self) :
        """
        """
        logging.info('get data statistics start')
        self.artist_gender_ = {}
        self.artist_date_action_ = [{}, {}, {}]
        self.artist_date_new_songs_ = {}
        for row in xrange(self.songs_.shape[0]) :
            entry = self.songs_.iloc[row,:]
            artist_date = str(entry.artist_id) + str(entry.publish_time)
            self.artist_date_new_songs_.setdefault(artist_date, 0)
            self.artist_date_new_songs_[artist_date] += 1

            self.artist_gender_[entry.artist_id] = entry.Gender

        for row in xrange(self.train_.shape[0]) :
            entry = self.train_.iloc[row,:]
            artist_date = str(entry.artist_id) + str(entry.Ds)
            self.artist_date_action_[int (entry.action_type) - 1].setdefault(artist_date, 0)

            self.artist_date_action_[int (entry.action_type) - 1][artist_date] += 1
            self.artist_gender_[entry.artist_id] = entry.Gender

        for row in xrange(self.test_.shape[0]) :
            entry = self.test_.iloc[row,:]
            artist_date = str(entry.artist_id) + str(entry.Ds)
            self.artist_date_action_[int (entry.action_type) - 1].setdefault(artist_date, 0)

            self.artist_date_action_[int (entry.action_type) - 1][artist_date] += 1
            self.artist_gender_[entry.artist_id] = entry.Gender

    def GetTrainingSet(self, consecutive = 3, gap = 1) :
        """
        """
        self.consecutive_ = consecutive
        self.gap_ = gap
        data = []
        label = []
        oneday = datetime.timedelta(days=1)
        conday = datetime.timedelta(days=self.consecutive_-1)
        gapday = datetime.timedelta(days=self.gap_)
        start_day = self.train_time_period_[0]
        columns_name = []
        ope = ['play', 'download', 'collect']
        columns_name.append ('Gender')
        for day in xrange(1, self.consecutive_ + 1) :
            for i in xrange(3) :
                columns_name.append (str (day) + '_' + ope[i])
            columns_name.append(str(day) + '_pub')

        while True:
            end_day = start_day + conday
            label_day = end_day + gapday
            if label_day > self.train_time_period_[1] :
                break
                
            for artist in self.artist_set_ :
                entry = []
                entry.append (self.artist_gender_[artist])
                today = start_day
                for day in xrange(self.consecutive_) :
                    for action in xrange(3) :
                        entry.append(self.artist_date_action_[action].get(artist + DateToStr(today), 0)) 
                    entry.append (self.artist_date_new_songs_.get(artist + DateToStr(today), 0))
                    today = today + oneday

                label.append([artist, self.artist_date_action_[0].get(artist + DateToStr(label_day), 0)])
                data.append(entry)
            start_day = start_day + oneday

        self.train_x_ = pd.DataFrame(data, columns = columns_name)
        self.train_y_ = pd.DataFrame(label, columns = ['artist_id', 'label'])
        logging.info('get training data done! the shape of data is (%d %d)' %self.train_x_.shape)

    def GetValidationSet(self, days = 1) :
        """
        """
        data = []
        label = []
        oneday = datetime.timedelta(days = 1)
        label_day = self.test_time_period_[0]
        columns_name = []
        ope = ['play', 'download', 'collect']
        columns_name.append ('Gender')
        for day in xrange(1, self.consecutive_ + 1) :
            for i in xrange(3) :
                columns_name.append (str (day) + '_' + ope[i])
            columns_name.append(str(day) + '_pub')

        for day in xrange(days) :
            for artist in self.artist_set_ :
                entry = []
                entry.append (self.artist_gender_[artist])
                today = label_day - datetime.timedelta(days = self.gap_ + self.consecutive_ - 1) 
                for day in xrange(self.consecutive_) :
                    for action in xrange(3) :
                        entry.append(self.artist_date_action_[action].get(artist + DateToStr(today), 0)) 
                    entry.append (self.artist_date_new_songs_.get(artist + DateToStr(today), 0))
                    today = today + oneday

                label.append([artist, self.artist_date_action_[0].get(artist + DateToStr(label_day), 0)])
                data.append(entry)

            label_day = label_day + oneday
            
        self.val_x_ = pd.DataFrame(data, columns = columns_name)
        self.val_y_ = pd.DataFrame(label, columns = ['artist_id', 'label'])
        logging.info('get validation data done! the shape of data is (%d %d)' %self.val_x_.shape)

    def FeatureHandler(self,) :
        """
        """
        self.train_x_ = binary_feature(self.train_x_, 'Gender')
        self.train_x_.drop('Gender', axis = 1, inplace = 1)
        self.val_x_ = binary_feature(self.val_x_, 'Gender')
        self.val_x_.drop('Gender', axis = 1, inplace = 1)
        
if __name__ == '__main__' :
    data = DataSet()
    

