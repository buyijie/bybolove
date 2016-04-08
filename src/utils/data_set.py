#!/usr/bin/env python
# coding=utf-8

import time
import os
import math
import shutil
import numpy as np
import pandas as pd
import logging
import datetime
import calendar
import threading
from multiprocessing import Process, Queue, Lock
import feature_handler

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
        # self.GetTimePeriod()
        self.GetFeature()

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
        self.data_ = pd.merge(self.action_, self.songs_, how = 'left', on = 'song_id')
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

    def SplitByMonth(self) :
        """
        """
        logging.info('start spliting data by month')
        self.data_ = self.data_.sort_values(['Ds'], ascending = True)
        self.month_data_ = []
        self.month_name_ = []
        st = 0
        month = self.data_.iloc[0,:].Ds[:6]
        block = max(1, int(math.sqrt(self.data_.shape[0] * 1.0)))
        for i in xrange(0, self.data_.shape[0], block) :
            if self.data_.iloc[i,:].Ds[:6] != month :
                ed = i - 1
                while True :
                    if self.data_.iloc[ed,:].Ds[:6] == month: break
                    ed -= 1
                self.month_name_.append(month)
                self.month_data_.append(self.data_[st:ed + 1])
                self.month_data_[-1] = self.month_data_[-1].sort_values(['user_id', 'song_id', 'Ds'], ascending = True)
                logging.info('for the month %s: from %d to %d, total %d' % (month, st, ed, ed - st + 1))
                st = ed + 1 
                month = self.data_.iloc[i,:].Ds[:6]
        self.month_name_.append(month)
        self.month_data_.append(self.data_[st:])
        self.month_data_[-1] = self.month_data_[-1].sort_values(['user_id', 'song_id', 'Ds'], ascending = True)
        logging.info('for the month %s: from %d to final, total %d' % (month, st, self.month_data_[-1].shape[0]))

    def GetSameUserSongPair(self, month, st) :
        """
        """
        n = self.month_data_[month].shape[0]
        ed = st 
        entry_st = self.month_data_[month].iloc[st,:]
        while ed < n :
            entry_ed = self.month_data_[month].iloc[ed,:]
            if entry_st.user_id + entry_st.song_id == entry_ed.user_id + entry_ed.song_id :
                ed += 1
            else : break
        return ed
    
    def GetSameUserSongDateTriple(self, month, st) :
        """
        """
        n = self.month_data_[month].shape[0]
        ed = st 
        entry_st = self.month_data_[month].iloc[st,:]
        while ed < n :
            entry_ed = self.month_data_[month].iloc[ed,:]
            if entry_st.user_id + entry_st.song_id + entry_st.Ds == entry_ed.user_id + entry_ed.song_id + entry_ed.Ds :
                ed += 1
            else : break
        return ed

    def GetArtistGender(self, sub) :
        """
        """
        return sub.iloc[0,:].Gender

    def GetSongLanguage(self, sub) :
        """
        """
        return sub.iloc[0,:].Language

    def GetPublishedDays(self, sub) :
        """
        """
        # from the publish time to 1st day of this month 
        today = StrToDate(sub.iloc[0,:].Ds)
        return (today - StrToDate(sub.iloc[0,:].publish_time)).days - today.day 

    def GetArtistID(self, sub) :
        """
        """
        return sub.iloc[0,:].artist_id
    
    def GetIsCollect(self, sub) :
        """
        """
        n = sub.shape[0]
        is_collect = 0
        for pre in xrange(n) :
            entry = sub.iloc[pre,:]
            if entry.action_type == 3:
                is_collect = 1
                break
        return is_collect
    
    def GetIsDownload(self, sub) :
        """
        """
        n = sub.shape[0]
        is_download = 0
        for pre in xrange(n) :
            entry = sub.iloc[pre,:]
            if entry.action_type == 2:
                is_download = 1
        return is_download

    def GetTotalPlaysForFeature(self, sub) :
        """
        """
        n = sub.shape[0]
        total_plays = 0
        for pre in xrange(n) :
            entry = sub.iloc[pre,:]
            if entry.action_type == 1:
                total_plays += 1
        return total_plays 

    def GetTotalPlaysForLabel(self, month, today, sub) :
        """
        """
        n = sub.shape[0]
        total_plays = 0
        for pre in xrange(n) :
            entry = sub.iloc[pre,:]
            if entry.action_type == 1:
                total_plays += 1
        return total_plays 

    def GetWeekday(self, month, today, sub) :
        """
        """
        weekday = today.weekday()
        return weekday

    def GetDay(self, month, today, sub) :
        """
        """
        return today.day
    
    def SingleFeatureProcess(self, id, month, function, st, ed, file_path, consecutive_days = None):
        """
        """
        logging.info("process %s start!" % id)
        user_song_st, user_song_ed = st, st 
        while user_song_st < ed :
            user_song_ed = self.GetSameUserSongPair(month, user_song_st)
            n = user_song_ed - user_song_st
            sub = self.month_data_[month].iloc[user_song_st:user_song_ed,]
            entry = sub.iloc[0,:]
            if user_song_st / 1000 != user_song_ed / 1000 :
                logging.info('process %s: handering %d samples!' % (id, user_song_st - st + 1))
            
            if consecutive_days == None :
                value = function(sub)
            else :
                lastday = calendar.monthrange(int(self.month_name_[month][:4]), int(self.month_name_[month][4:]))[1]
                begin_day = StrToDate(self.month_name_[month][:4] + str(lastday - consecutive_days + 1))
                
                user_song_date_st = 0
                while user_song_date_st < n and StrToDate(sub.iloc[user_song_date_st,:].Ds) < begin_day :
                    user_song_date_st += 1
                value = function(sub[user_song_date_st:])
                
            if value != None:
                with open(file_path, 'a') as out :
                    out.write(entry.user_id + '#' + entry.song_id + ',' + str(value) + '\n')

            user_song_st = user_song_ed

    def LabelProcess(self, id, month, function, st, ed, file_path, consecutive_days = None):
        """
        """
        logging.info("process %s start!" % id)
        user_song_st, user_song_ed = st, st 
        one_day = datetime.timedelta(days = 1)
        while user_song_st < ed :
            user_song_ed = self.GetSameUserSongPair(month, user_song_st)
            n = user_song_ed - user_song_st
            sub = self.month_data_[month].iloc[user_song_st:user_song_ed,]
            entry = sub.iloc[0,:]
            if user_song_st / 1000 != user_song_ed / 1000 :
                logging.info('process %s: handering %d samples!' % (id, user_song_st - st + 1))

            user_song_date_st = 0
            begin_day = StrToDate(self.month_name_[month] + '01') 
            today = begin_day
            while today.month == begin_day.month:
                if user_song_date_st < n and DateToStr(today) == sub.iloc[user_song_date_st,:].Ds :
                    user_song_date_ed = self.GetSameUserSongDateTriple(month, user_song_date_st)
                    value = function(month, today, sub.iloc[user_song_date_st:user_song_date_ed])
                    user_song_date_st = user_song_date_ed
                else:
                    value = function(month, today, sub.iloc[0:0])

                if value != None:
                    with open(file_path, 'a') as out :
                        out.write(entry.user_id + '#' + entry.song_id + ',' + str(value) + '\n')
                today = today + one_day
 
            user_song_st = user_song_ed

    def GetFeatureInOneMonth(self, month, feature_name, extract_function, process_function, consecutive_days = None) :
        """
        """
        logging.info('get feature %s in month %s' %(feature_name, self.month_name_[month]))
        filepath = ROOT + '/data/' + feature_name + '_' + self.month_name_[month] + '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '_' + str(self.gap_month_) + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        n = self.month_data_[month].shape[0]

        seperate = [0]
        for i in xrange(1, self.n_jobs_) :
            point = n / self.n_jobs_ * i
            point = self.GetSameUserSongPair(month, point)
            seperate.append(point)
        seperate.append(n)

        processes = []
        for i in xrange(self.n_jobs_) :
            process = Process(target = process_function, args = (str(i + 1), month, extract_function, seperate[i], seperate[i + 1], filepath[:-4] + str(i) + filepath[-4:], consecutive_days))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

        with open (filepath, 'w') as out :
            out.write ('user_song,' + feature_name + '\n')
        for i in xrange(self.n_jobs_) :
            os.system("cat " + filepath[:-4] + str(i) + filepath[-4:] + " >> " + filepath)
            os.remove(filepath[:-4] + str(i) + filepath[-4:])

        logging.info('the feature %s is write into %s' % (feature_name, filepath))

    def GetSingleFeature(self, feature_name, function, consecutive_days = None) :
        """
        """
        logging.info('get feature: %s' % feature_name)
        for month in xrange(len(self.month_data_)) :
            self.GetFeatureInOneMonth(month, feature_name, function, self.SingleFeatureProcess, consecutive_days)

    def GetLabel(self, label_name, function) :
        """
        """
        logging.info('get label: %s' % label_name)
        for month in xrange(len(self.month_data_)) :
            self.GetFeatureInOneMonth(month, label_name, function, self.LabelProcess)

    def GetFeature(self, consecutive_recent = [14, 7, 3], gap_month = 1, gap_day = 0) :
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
        self.consecutive_recent_ = consecutive_recent
        self.gap_month_ = gap_month
        self.gap_day_ = gap_day

        self.SplitByMonth()

        self.GetSingleFeature('gender_of_artist', self.GetArtistGender)
        self.GetSingleFeature('language_of_song', self.GetSongLanguage)
        self.GetSingleFeature('published_days', self.GetPublishedDays)
        self.GetSingleFeature('artist_id', self.GetArtistID)

        self.GetSingleFeature('total_plays_for_one_song_all', self.GetTotalPlaysForFeature)
        for consecutive_days in self.consecutive_recent_:
            self.GetSingleFeature('total_plays_for_one_song_recent_' + str(consecutive_days), self.GetTotalPlaysForFeature, consecutive_days)

        self.GetSingleFeature('is_collect', self.GetIsCollect)
        self.GetSingleFeature('is_download', self.GetIsDownload)

        self.GetLabel('label_plays', self.GetTotalPlaysForLabel)
        self.GetLabel('label_weekday', self.GetWeekday)
        self.GetLabel('label_day', self.GetDay)

    def GetFromFile(self, month, feature_name) :
        """
        """
        logging.info('get feature %s in month %s from file' %(feature_name, self.month_name_[month]))
        filepath = ROOT + '/data/' + feature_name + '_' + self.month_name_[month] + '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '_' + str(self.gap_month_) + '.csv'
        if not os.path.exists(filepath) :
            os.system('cat ' + filepath)
            logging.error(filepath + ' doesn\'t exists!')
            exit(1)

        data = pd.read_csv(filepath)
        return data

    def GetSongInfoForMergingData(self, song, feature) :
        """
        """
        if feature == 'gender_of_artist' : feature = 'Gender'
        if feature == 'language_of_song' : feature = 'Language'
        return self.songs_.loc[self.songs_.song_id == song, feature].values.tolist()[0]

    def MergeData(self, x, y, y_month) :
        """
        merge the feature and label by user_song
        """
        data = pd.merge(x, y, how = 'outer', on = 'user_song')
        data.sort_values(['user_song', 'label_day'])
        
        # how many days in y_month
        days = calendar.monthrange(int(self.month_name_[y_month][:4]), int(self.month_name_[y_month][4:]))[1]
        # the weekday for 1st
        weekday_1st = StrToDate(self.month_name_[y_month] + '01').weekday()

        # y exist, x not exitst
        data['song_id'] = data.user_song.map(lambda v : v.split('#')[1])
        feature_list = ['gender_of_artist', 'artist_id', 'language_of_song']
        for feature in feature_list :
            groups = data[data[feature].isnull()].groupby(['song_id']).groups
            for name, group in groups.items() :
                data.loc[group, feature] = self.GetSongInfoForMergingData(name, feature)

            # data.loc[data[feature].isnull(), feature] = data.loc[data[feature].isnull(), 'user_song'].map(lambda v : self.GetSongInfoForMergingData(v, feature))
        data.drop('song_id', inplace = True, axis = 1)
        
        # x exist, y not exist
        table = data[data.label_day.isnull()] 
        data = data[data.label_day.isnull() == False]

        for i in xrange(days) :
            day = i + 1
            weekday = (weekday_1st + i) % 7

            table['label_weekday'] = weekday
            table['label_day'] = day

            data = pd.concat([data, table])

        return data

    def GetData(self, month_for_test = 2) :
        """
        """
        feature_list = ['artist_id', 'gender_of_artist', 'language_of_song', 'published_days', 'total_plays_for_one_song_all', 'is_collect', 'is_download']
        for consecutive_days in self.consecutive_recent_ :
            feature_list.append('total_plays_for_one_song_recent_' + str(consecutive_days))
        label_list = ['label_plays', 'label_day', 'label_weekday']

        self.final_data_ = None
        first_month = True
        cnt_month = []
        for month in  xrange(len(self.month_name_) - self.gap_month_) :
            first = True
            y_data = None 
            for label in label_list :
                data = self.GetFromFile(month + self.gap_month_, label)
                if first : y_data = data
                else : y_data[label] = data[label]
                first = False

            first = True
            x_data = []
            for feature in feature_list:
                data = self.GetFromFile(month, feature)
                if first : x_data = data
                else : x_data[feature] = data[feature]
                first = False

            final_data = self.MergeData(x_data, y_data, month + self.gap_month_)

            cnt_month.append(final_data.shape[0])
            if first_month:
                self.final_data_ = final_data
            else :
                self.final_data_ = pd.concat([self.final_data_, final_data])
            first_month = False

        # replace the null value by zero
        for label in label_list:
            self.final_data_[label][self.final_data_[label].isnull()] = 0
        for feature in feature_list:
            self.final_data_[feature][self.final_data_[feature].isnull()] = 0

        # binary feature
        binary_feature = ['gender_of_artist', 'language_of_song', 'label_day', 'label_weekday']
        for feature in binary_feature:
            self.final_data_ = feature_handler.binary_feature(self.final_data_, feature)

        testing = sum(cnt_month[-month_for_test:])
        return self.final_data_[:-testing], self.final_data_[-testing:]


if __name__ == '__main__' :
    data = DataSet()
