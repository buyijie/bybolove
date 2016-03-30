#!/usr/bin/env python
# coding=utf-8

import os
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

def StrToDate(str) :
    """
    """
    return datetime.datetime.strptime(str, "%Y%m%d")

class DataSet :
    """
    """
    def __init__ (self, type = "unit") :
        self.type_ = type
        self.Read()
        self.Join()
        self.GetTimePeriod()
        self.GetData()

    def Read(self) :
        """
        read data from csv file
        """
        logging.info('read data from csv file, the type is %s'%self.type_)
        if self.type_ == "unit" :
            self.songs_ = pd.read_csv(ROOT + '/data/mars_tianchi_songs_tiny.csv', header = None)
            self.action_ = pd.read_csv(ROOT + '/data/mars_tianchi_user_actions_tiny.csv', header = None)
            self.label_ = pd.read_csv(ROOT + '/data/label_tiny.csv', header = None)
        elif self.type_ == 'full' :
            self.songs_ = pd.read_csv(ROOT + '/data/mars_tianchi_songs.csv', header = None)
            self.action_ = pd.read_csv(ROOT + '/data/mars_tianchi_user_actions.csv', header = None)
            self.label_ = pd.read_csv(ROOT + '/data/label.csv', header = None)
        else:
            logging.error('Invalid type of data set, please choose unit or full')
            exit (1)

        self.songs_.columns = ['song_id', 'artist_id', 'publish_time', 'song_init_plays', 'Language', 'Gender']
        self.action_.columns = ['user_id', 'song_id', 'gmt_create', 'action_type', 'Ds']
        self.label_.columns = ['artist_id', 'Plays', 'Ds']

    def Join(self) :
        """
        """
        logging.info("merge action and songs by song_id")
        self.data_ = pd.merge(self.action_, self.songs_, on = 'song_id')
        self.data_['Ds'] = self.data_['Ds'].map(lambda v: str(v))
        print self.data_.shape

    def GetTimePeriod(self) :
        """
        get the time period in training set and validation set
        """
        logging.info('get time period')
        time_period = sorted(map(lambda v: StrToDate(v), set(self.data_.Ds.values)))
        self.time_period_ = [time_period[0], time_period[-1]] 
        logging.info('the time period: %s to %s'%(time_period[0].strftime("%Y-%m-%d"), time_period[-1].strftime("%Y-%m-%d")))

    def GetSamePair(self, st) :
        """
        """
        n = self.user_song_date_sorted_.shape[0]
        ed = st 
        entry_st = self.user_song_date_sorted_.iloc[st,:]
        while ed < n :
            entry_ed = self.user_song_date_sorted_.iloc[ed,:]
            if entry_st.user_id + entry_st.song_id == entry_ed.user_id + entry_ed.song_id :
                ed += 1
            else : break
        return ed

    def GetArtistGender(self, entry, today) :
        """
        """
        return entry.Gender

    def GetSongLanguage(self, entry, today) :
        """
        """
        return entry.Language

    def GetPublishedDays(self, entry, today) :
        """
        """
        return (today - entry.publish_time).days

    def GetSingleFeature(self, feature_name, function) :
        """
        """
        logging.info('get the ' + feature_name)
        filepath = ROOT + '/data/' + feature_name + '_' + str(self.consecutive_all_) + '_' + str(self.consecutive_last_) + '_' + str(self.gap_) + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        feature = []
        n = self.user_song_date_sorted_.shape[0]
        st, ed = 0, 0
        con_day = datetime.timedelta(days = self.consecutive_all_ - 1)
        gap_day = datetime.timedelta(days = self.gap_)
        while st < n :
            ed = self.GetSamePair(st)
            sub = self.user_song_date_sorted_.iloc[st:ed,]
            m = ed - st 
            if st / 10000 != ed / 10000 :
                logging.info('handering %d samples!' % st)
            st = ed

            for index in xrange(m) :
                today = StrToDate(sub.iloc[index,:].Ds)
                end_day = today - gap_day
                begin_day = end_day - con_day
                is_exist = False

                for pre in xrange(index - 1 , -1 , -1) :
                    entry = sub.iloc[pre,:]
                    this_day = StrToDate(entry.Ds)
                    if this_day <= end_day :
                        if this_day >= begin_day :
                            is_exist = True
                        break
                
                if is_exist:
                    feature.append([entry.user_id + entry.song_id + DateToStr(today), function(entry, today)])

        feature_csv = pd.DataFrame(feature, columns = ['user_song_date', feature_name])
        feature_csv.to_csv(filepath, encoding = 'utf-8', index = False)

    def GetData(self, consecutive_all = 30, consecutive_last = 3, gap = 1, training_proportion = 0.8) :
        """
        feature list:
        1. gender of artist 
        2. language of song
        3. how many days have been published for this song 
        4. the number of total plays for current user and current song in consecutive_all days 
        5. the number of total plays for current user and current song in consecutive_last days 
        6. the number of total plays for current user and all the song in consecutive_all days 
        7. the number of total plays for current user and all the song in consecutive_last days 
        8. the proportion of the songs that the current user plays in consecutive_all days 
        9. the proportion of the songs that the currect user plays in consecutive_last days 
        10. the proportion of the artist that the current user plays in consecutive_all days 
        11. the proportion of the artist that the currect user plays in consecutive_last days 
        12. whether the current user have collected this song 
        13. whether the current user have downloaded this song
        """
        logging.info("start generating data according to feature list") 
        self.consecutive_all_ = consecutive_all
        self.consecutive_last_ = consecutive_last
        self.gap_ = gap
        self.training_proportion_ = training_proportion
        logging.info("the number of feature is %d" % self.kFeature)

        self.user_song_date_ = {}
    
        self.user_song_date_sorted_ = self.data_.sort_values(['user_id', 'song_id', 'Ds'], ascending = True)
        self.GetSingleFeature('gender_of_artist', self.GetArtistGender)
        self.GetSingleFeature('language_of_song', self.GetSongLanguage)
        self.GetSingleFeature('published_days', self.GetPublishedDays)

        
        
        


if __name__ == '__main__' :
    data = DataSet()
