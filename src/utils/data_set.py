#!/usr/bin/env python
# coding=utf-8

import os
import shutil
import numpy as np
import pandas as pd
import logging
import datetime
import threading
from multiprocessing import Process, Queue, Lock
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
    def __init__ (self, type = "unit", n_jobs = 1) :
        self.type_ = type
        self.n_jobs_ = n_jobs
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
        self.data_['publish_time'] = self.data_['publish_time'].map(lambda v: str(v))
        print self.data_.shape

    def GetTimePeriod(self) :
        """
        get the time period in training set and validation set
        """
        logging.info('get time period')
        time_period = sorted(map(lambda v: StrToDate(v), set(self.data_.Ds.values)))
        self.time_period_ = [time_period[0], time_period[-1]] 
        logging.info('the time period: %s to %s'%(time_period[0].strftime("%Y-%m-%d"), time_period[-1].strftime("%Y-%m-%d")))

    def GetSameUserSongPair(self, st) :
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
    
    def GetSameUserSongDateTriple(self, st, data) :
        """
        """
        n = data.shape[0]
        ed = st 
        entry_st = data.iloc[st,:]
        while ed < n :
            entry_ed = data.iloc[ed,:]
            if entry_st.user_id + entry_st.song_id + entry_st.Ds == entry_ed.user_id + entry_ed.song_id + entry_ed.Ds :
                ed += 1
            else : break
        return ed

    def GetArtistGender(self, begin_day, end_day, sub) :
        """
        """
        n = sub.shape[0]
        is_exist = False
        for pre in xrange(n - 1 , -1 , -1) :
            entry = sub.iloc[pre,:]
            this_day = StrToDate(entry.Ds)
            if this_day <= end_day :
                if this_day >= begin_day :
                    is_exist = True
                break
        if not is_exist : 
            return None
        return entry.Gender

    def GetSongLanguage(self, begin_day, end_day, sub) :
        """
        """
        n = sub.shape[0]
        is_exist = False
        for pre in xrange(n - 1 , -1 , -1) :
            entry = sub.iloc[pre,:]
            this_day = StrToDate(entry.Ds)
            if this_day <= end_day :
                if this_day >= begin_day :
                    is_exist = True
                break
        if not is_exist : 
            return None
        return entry.Language

    def GetPublishedDays(self, begin_day, end_day, sub) :
        """
        """
        n = sub.shape[0]
        is_exist = False
        for pre in xrange(n - 1 , -1 , -1) :
            entry = sub.iloc[pre,:]
            this_day = StrToDate(entry.Ds)
            if this_day <= end_day :
                if this_day >= begin_day :
                    is_exist = True
                break
        if not is_exist : 
            return None
        return (begin_day - StrToDate(entry.publish_time)).days
    
    def GetIsCollect(self, begin_day, end_day, sub) :
        """
        """
        n = sub.shape[0]
        is_exist = False
        is_collect = 0
        for pre in xrange(n - 1 , -1 , -1) :
            entry = sub.iloc[pre,:]
            this_day = StrToDate(entry.Ds)
            if this_day <= end_day :
                if this_day >= begin_day :
                    is_exist = True
                    if entry.action_type == 3:
                        is_collect = 1
        if not is_exist : 
            return None
        return is_collect
    
    def GetIsDownload(self, begin_day, end_day, sub) :
        """
        """
        n = sub.shape[0]
        is_exist = False
        is_download = 0
        for pre in xrange(n - 1 , -1 , -1) :
            entry = sub.iloc[pre,:]
            this_day = StrToDate(entry.Ds)
            if this_day <= end_day :
                if this_day >= begin_day :
                    is_exist = True
                    if entry.action_type == 2:
                        is_download = 1
        if not is_exist : 
            return None
        return is_download

    def GetTotalPlays(self, begin_day, end_day, sub) :
        """
        """
        n = sub.shape[0]
        is_exist = False
        total_plays = 0
        for pre in xrange(n - 1 , -1 , -1) :
            entry = sub.iloc[pre,:]
            this_day = StrToDate(entry.Ds)
            if this_day <= end_day :
                if this_day >= begin_day :
                    is_exist = True
                    if entry.action_type == 1:
                        total_plays += 1
        if not is_exist : 
            return None
        return total_plays 
    
    def SingleFeatureProcess(self, id, consecutive_days, function, st, ed, file_path):
        """
        """
        logging.info("process %s start!" % id)
        user_song_st, user_song_ed = st, st 
        one_day = datetime.timedelta(days = 1)
        con_day = datetime.timedelta(days = consecutive_days - 1)
        gap_day = datetime.timedelta(days = self.gap_)
        while user_song_st < ed :
            user_song_ed = self.GetSameUserSongPair(user_song_st)
            sub = self.user_song_date_sorted_.iloc[user_song_st:user_song_ed,]
            entry = sub.iloc[0,:]
            m = user_song_ed - user_song_st 
            if user_song_st / 1000 != user_song_ed / 1000 :
                logging.info('process %s: handering %d samples!' % (id, user_song_st - st + 1))
            user_song_st = user_song_ed
            
            delta_days = (self.time_period_[1] - self.time_period_[0]).days + 1
            begin_day = self.time_period_[0]
            user_song_date_st = 0
            for day in xrange(delta_days - consecutive_days - self.gap_ + 1) :
                # x: [begin_day, end_day] y: target_day
                end_day = begin_day + con_day
                target_day = end_day + gap_day

                while user_song_date_st < m and StrToDate(sub.iloc[user_song_date_st,:].Ds) <= end_day :
                    user_song_date_st += 1

                value = function(begin_day, end_day, sub[0:user_song_date_st])
                begin_day = begin_day + one_day

                if value:
                    with open(file_path, 'a') as out :
                        out.write(entry.user_id + entry.song_id + DateToStr(target_day) + ',' + str(value) + '\n')

    def GetSingleFeature(self, feature_name, consecutive_days, function) :
        """
        """
        logging.info('get feature: ' + feature_name)
        filepath = ROOT + '/data/' + feature_name + '_' + self.type_ + '_' + str(self.consecutive_all_) + '_' + str(self.consecutive_last_) + '_' + str(self.gap_) + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        n = self.user_song_date_sorted_.shape[0]

        seperate = [0]
        for i in xrange(1, self.n_jobs_) :
            point = n / self.n_jobs_ * i
            point = self.GetSameUserSongPair(point)
            seperate.append(point)
        seperate.append(n)

        processes = []
        for i in xrange(self.n_jobs_) :
            process = Process(target = self.SingleFeatureProcess, args = (str(i + 1), consecutive_days, function, seperate[i], seperate[i + 1], filepath[:-4] + str(i) + filepath[-4:]))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

        with open (filepath, 'w') as out :
            out.write ('user_song_date,' + feature_name + '\n')
        for i in xrange(self.n_jobs_) :
            os.system("cat " + filepath[:-4] + str(i) + filepath[-4:] + " >> " + filepath)
            os.remove(filepath[:-4] + str(i) + filepath[-4:])

        logging.info('the feature %s is write into %s' % (feature_name, filepath))

    def GetLabel(self, label_name, consecutive_days, function) :
        """
        """
        logging.info('get label: ' + label_name)
        filepath = ROOT + '/data/' + label_name + '_' + self.type_ + '_' + str(self.consecutive_all_) + '_' + str(self.consecutive_last_) + '_' + str(self.gap_) + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        n = self.user_song_date_sorted_.shape[0]
       
        seperate = [0]
        for i in xrange(1, self.n_jobs_) :
            point = n / self.n_jobs_ * i
            point = self.GetSameUserSongPair(point)
            seperate.append(point)
        seperate.append(n)

        processes = []
        for i in xrange(self.n_jobs_) :
            process = Process(target = self.SingleFeatureProcess, args = (str(i + 1), consecutive_days, function, seperate[i], seperate[i + 1], filepath[:-4] + str(i) + filepath[-4:]))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

        with open (filepath, 'w') as out :
            out.write ('user_song_date,' + label_name + '\n')
        for i in xrange(self.n_jobs_) :
            os.system("cat " + filepath[:-4] + str(i) + filepath[-4:] + " >> " + filepath)
            os.remove(filepath[:-4] + str(i) + filepath[-4:])

        logging.info('the feature %s is write into %s' % (feature_name, filepath))

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

        self.user_song_date_ = {}
    
        self.user_song_date_sorted_ = self.data_.sort_values(['user_id', 'song_id', 'Ds'], ascending = True)

        #self.GetSingleFeature('gender_of_artist', self.consecutive_all_, self.GetArtistGender)
        #self.GetSingleFeature('language_of_song', self.consecutive_all_, self.GetSongLanguage)
        #self.GetSingleFeature('published_days', self.consecutive_all_, self.GetPublishedDays)

        #self.GetSingleFeature('total_plays_for_one_song_all', self.consecutive_all_, self.GetTotalPlays)
        #self.GetSingleFeature('total_plays_for_one_song_recent', self.consecutive_last_, self.GetTotalPlays)


        #self.GetSingleFeature('is_collect', self.consecutive_all_, self.GetIsCollect)
        #self.GetSingleFeature('is_download', self.consecutive_all_, self.GetIsDownload)


        self.GetLabel('label_plays', self.consecutive_all_, self.GetTotalPlays)

        
        
        


if __name__ == '__main__' :
    data = DataSet()
