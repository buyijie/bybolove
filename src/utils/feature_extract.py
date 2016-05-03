#!/usr/bin/env python
# coding=utf-8

import os
import math
import time
import pandas as pd
import pkl
import logging
import logging.config
import datetime
import calendar
import getopt
from multiprocessing import Process

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from configure import *

logging.config.fileConfig('../logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))

def DateToStr(date) :
    """
    """
    return date.strftime('%Y%m%d')

def StrToDate(str) :
    """
    """
    return datetime.datetime.strptime(str, "%Y%m%d")

class FeatureExtract:
    """
    """
    def __init__ (self, type = "unit", n_jobs = 1, consecutive_recent = [14, 7, 3],  gap_day = 0) :
        self.consecutive_recent_ = consecutive_recent
        self.gap_day_ = gap_day
        self.type_ = type
        self.n_jobs_ = n_jobs
        self.Read()
        self.Join()
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
        self.song_id_set_ = self.songs_.song_id.values.tolist().sort()
        self.artist_list_ = sorted(set(self.songs_['artist_id'])) 

    def Join(self) :
        """
        """
        logging.info("merge action and songs by song_id")
        self.data_ = pd.merge(self.action_, self.songs_, how = 'left', on = 'song_id')
        self.data_['Ds'] = self.data_['Ds'].map(lambda v: str(v))
        self.data_['publish_time'] = self.data_['publish_time'].map(lambda v: str(v))
        self.data_['gmt_create'] = self.data_['gmt_create'].map(lambda v : time.gmtime(int(v)).tm_hour)
        del self.action_
        logging.info('the size of whole data after joining is (%d %d)' % self.data_.shape)

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
                self.month_data_[-1] = self.month_data_[-1].sort_values(['song_id', 'Ds'], ascending = True)
                logging.info('for the month %s: from %d to %d, total %d' % (month, st, ed, ed - st + 1))
                st = ed + 1
                month = self.data_.iloc[i,:].Ds[:6]
        self.month_name_.append(month)
        pkl.store(self.month_name_, ROOT + '/data/month_name.pkl')
        self.month_data_.append(self.data_[st:])
        self.month_data_[-1] = self.month_data_[-1].sort_values(['song_id', 'Ds'], ascending = True)
        logging.info('for the month %s: from %d to final, total %d' % (month, st, self.month_data_[-1].shape[0]))

    def GetSameSong(self, month, st) :
        """
        """
        n = self.month_data_[month].shape[0]
        ed = st
        entry_st = self.month_data_[month].iloc[st,:]
        while ed < n :
            entry_ed = self.month_data_[month].iloc[ed,:]
            if entry_st.song_id == entry_ed.song_id :
                ed += 1
            else : break
        return ed

    def GetSameSongDatePair(self, month, st) :
        """
        """
        n = self.month_data_[month].shape[0]
        ed = st
        entry_st = self.month_data_[month].iloc[st,:]
        while ed < n :
            entry_ed = self.month_data_[month].iloc[ed,:]
            if entry_st.song_id + entry_st.Ds == entry_ed.song_id + entry_ed.Ds :
                ed += 1
            else : break
        return ed

    def GetArtistGender(self, sub, condition_hour = None) :
        """
        """
        return sub.iloc[0,:].Gender

    def GetSongLanguage(self, sub, condition = None) :
        """
        """
        return sub.iloc[0,:].Language

    def GetPublishedDays(self, sub, condition_hour = None) :
        """
        """
        # from the publish time to 1st day of this month
        today = StrToDate(sub.iloc[0,:].Ds)
        return (today - StrToDate(sub.iloc[0,:].publish_time)).days - today.day

    def GetArtistID(self, sub, condition_hour = None) :
        """
        """
        return sub.iloc[0,:].artist_id

    def GetSongInitPlays(self, sub, condition_hour = None) :
        """
        """
        return sub.iloc[0,:].song_init_plays

    def GetIsCollect(self, sub, condition_hour = None) :
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

    def GetIsDownload(self, sub, condition_hour = None) :
        """
        """
        n = sub.shape[0]
        is_download = 0
        for pre in xrange(n) :
            entry = sub.iloc[pre,:]
            if entry.action_type == 2:
                is_download = 1
        return is_download

    def GetTotalPlaysForFeature(self, sub, condition_hour = None) :
        """
        """
        return len(sub.groupby(['action_type']).groups.get(1, []))

    def GetTotalPlaysForFeatureInSpecificHour(self, sub, condition_hour) :
        """
        """
        return len(sub.groupby(['action_type', 'gmt_create']).groups.get((1, condition_hour), []))
    
    def GetTotalPlaysForFeatureInHourInterval(self, sub, condition_hour_interval) :
        """
        """
        ret=0
        for hour in condition_hour_interval:
            ret+=self.GetTotalPlaysForFeatureInSpecificHour(sub,hour)
        return ret

    def GetTotalPlaysForLabel(self, month, today, sub) :
        """
        """
        return len(sub.groupby(['action_type']).groups.get(1, []))

    def GetWeekday(self, month, today, sub) :
        """
        """
        weekday = today.weekday()
        return weekday

    def GetDay(self, month, today, sub) :
        """
        """
        return today.day

    def SingleFeatureProcess(self, id, month, function, st, ed, file_path, consecutive_days = None, condition_hour = None):
        """
        """
        logging.info("process %s start!" % id)
        song_st, song_ed = st, st
        while song_st < ed :
            song_ed = self.GetSameSong(month, song_st)
            n = song_ed - song_st
            sub = self.month_data_[month].iloc[song_st:song_ed,]
            entry = sub.iloc[0,:]
            if song_st / 1000 != song_ed / 1000 :
                logging.info('process %s: handering %d samples!' % (id, song_st - st + 1))

            if consecutive_days == None :
                value = function(sub, condition_hour)
            else :
                lastday = calendar.monthrange(int(self.month_name_[month][:4]), int(self.month_name_[month][4:]))[1]
                begin_day = StrToDate(self.month_name_[month][:4] + str(lastday - consecutive_days + 1))

                song_date_st = 0
                while song_date_st < n and StrToDate(sub.iloc[song_date_st,:].Ds) < begin_day :
                    song_date_st += 1
                value = function(sub[song_date_st:], condition_hour)

            if value != None:
                with open(file_path, 'a') as out :
                    out.write(entry.song_id + ',' + str(value) + '\n')

            song_st = song_ed

    def LabelProcess(self, id, month, function, st, ed, file_path, consecutive_days = None, condition_hour = None):
        """
        """
        logging.info("process %s start!" % id)
        song_st, song_ed = st, st
        one_day = datetime.timedelta(days = 1)
        while song_st < ed :
            song_ed = self.GetSameSong(month, song_st)
            n = song_ed - song_st
            sub = self.month_data_[month].iloc[song_st:song_ed,]
            entry = sub.iloc[0,:]
            if song_st / 1000 != song_ed / 1000 :
                logging.info('process %s: handering %d samples!' % (id, song_st - st + 1))

            song_date_st = song_st 
            begin_day = StrToDate(self.month_name_[month] + '01')
            today = begin_day
            while today.month == begin_day.month:
                if song_date_st < ed and DateToStr(today) == self.month_data_[month].iloc[song_date_st,:].Ds :
                    song_date_ed = self.GetSameSongDatePair(month, song_date_st)
                    value = function(month, today, self.month_data_[month].iloc[song_date_st:song_date_ed])
                    song_date_st = song_date_ed
                else:
                    value = function(month, today, sub.iloc[0:0])

                if value != None:
                    with open(file_path, 'a') as out :
                        out.write(entry.song_id + ',' + str(value) + '\n')
                today = today + one_day

            song_st = song_ed

    def GetFeatureInOneMonth(self, month, feature_name, extract_function, process_function, consecutive_days = None, condition_hour = None) :
        """
        """
        logging.info('get feature %s in month %s' %(feature_name, self.month_name_[month]))
        filepath = ROOT + '/data/' + feature_name + '_' + self.month_name_[month] + '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        n = self.month_data_[month].shape[0]

        seperate = [0]
        for i in xrange(1, self.n_jobs_) :
            point = n / self.n_jobs_ * i
            point = self.GetSameSong(month, point)
            seperate.append(point)
        seperate.append(n)

        processes = []
        for i in xrange(self.n_jobs_) :
            process = Process(target = process_function, args = (str(i + 1), month, extract_function, seperate[i], seperate[i + 1], filepath[:-4] + str(i) + filepath[-4:], consecutive_days, condition_hour))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

        with open (filepath, 'w') as out :
            out.write ('song_id,' + feature_name + '\n')
        for i in xrange(self.n_jobs_) :
            if not os.path.exists(filepath[:-4] + str(i) + filepath[-4:]) :
                continue
            os.system("cat " + filepath[:-4] + str(i) + filepath[-4:] + " >> " + filepath)
            os.remove(filepath[:-4] + str(i) + filepath[-4:])

        logging.info('the feature %s is write into %s' % (feature_name, filepath))

    def GetSingleFeature(self, feature_name, function, consecutive_days = None, condition_hour = None) :
        """
        """
        logging.info('get feature: %s' % feature_name)
        for month in xrange(len(self.month_data_)) :
            self.GetFeatureInOneMonth(month, feature_name, function, self.SingleFeatureProcess, consecutive_days, condition_hour)

    def GetLabel(self, label_name, function) :
        """
        """
        logging.info('get label: %s' % label_name)
        for month in xrange(len(self.month_data_)) :
            self.GetFeatureInOneMonth(month, label_name, function, self.LabelProcess)
    
    def GetFromFile(self,feature_name, month) :
        """
        """
        filepath = ROOT + '/data/' + feature_name + '_' + self.month_name_[month] + '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '.csv'
        if not os.path.exists(filepath) :
            logging.error(filepath + ' doesn\'t exists!')
            exit(-1)
        return pd.read_csv(filepath)

    def GetTotalPlaysForArtistInOneMonth(self, feature_name, base_feature, feature_for_artist ,month) :
        """
        """
        logging.info('get feature for %s in month % s' % (feature_name, self.month_name_[month]))
        filepath = ROOT + '/data/' + feature_name + '_' + self.month_name_[month] + '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        base = self.GetFromFile(base_feature, month)
        artist = self.GetFromFile(feature_for_artist, month)
        data = pd.merge(base, artist, how = 'left', on = 'song_id')
        data[feature_name] = data[base_feature].groupby(data[feature_for_artist]).transform('sum')
        data.drop(base_feature, inplace = True, axis = 1)
        data.drop(feature_for_artist, inplace = True, axis = 1)
        data.to_csv(filepath, index = False)
        logging.info('the feature %s is write into %s' % (feature_name, filepath))

    def GetTotalPlaysForArtist(self, feature_name, base_feature, feature_for_artist) :
        """
        """
        logging.info('used %s to generate %s'% (base_feature, feature_name))
        for month in xrange(len(self.month_data_)) :
            self.GetTotalPlaysForArtistInOneMonth(feature_name, base_feature, feature_for_artist, month)

    def GetTotalPlaysForEveryArtistInOneMonth(self, feature_name, base_feature, feature_for_artist, month) :
        """
        """
        logging.info('get feature for %s in month %s' % (feature_name, self.month_name_[month]))
        base = self.GetFromFile(base_feature, month)
        artist = self.GetFromFile(feature_for_artist, month)
        data = pd.merge(base, artist, how = 'left', on = 'song_id')       
        group = data[base_feature].groupby(data[feature_for_artist]).groups

        for idx in xrange(len(self.artist_list_)):
            filepath = ROOT + '/data/' + feature_name+'_'+str(idx)+ '_' + self.month_name_[month] + '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '.csv'
            if os.path.exists(filepath) :
                logging.info(filepath + ' exists!')
                continue 
            data[feature_name+'_'+str(idx)] = 0 if group.get(self.artist_list_[idx])==None else sum(group.get(self.artist_list_[idx]))
            data.to_csv(filepath, columns=[feature_name+'_'+str(idx)] ,index = False)
            logging.info('the feature %s is write into %s' % (feature_name+'_'+str(idx), filepath))
           

    def GetTotalPlaysForEveryArtist(self, feature_name, base_feature, feature_for_artist) :
        """
        """
        logging.info('used %s to generate %s'% (base_feature, feature_name))
        for month in xrange(len(self.month_data_)) :
            self.GetTotalPlaysForEveryArtistInOneMonth(feature_name, base_feature, feature_for_artist, month)

    def GetCombinationFeatureInOneMonth(self, A, B, month, ope) :
        """
        """
        feature_name = A + '_' + ope + '_' + B
        logging.info('get combination feature from %s and %s by %s in month % s' % (A, B, ope, self.month_name_[month]))
        filepath = ROOT + '/data/' + feature_name + '_' + self.month_name_[month] + '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        data_a = self.GetFromFile(A, month)
        data_b = self.GetFromFile(B, month)
        data = pd.merge(data_a, data_b, on = 'song_id')
        if ope == 'div' :
            data[feature_name] = data[A] * 1.0 / data[B]
        elif ope == 'mul' :
            data[feature_name] = data[A] * data[B]
        else :
            logging.error('the operation of %s is invalid' % ope)
            exit(-1)
        data.drop(A, axis = 1, inplace = True)
        data.drop(B, axis = 1, inplace = True)
        data.to_csv(filepath, index = False)
        logging.info('the feature %s is write into %s' % (feature_name, filepath))

    def GetCombinationFeature(self, A, B, ope) :
        """
        """
        logging.info('get combination feature from %s and %s by %s'% (A, B, ope))
        for month in xrange(len(self.month_data_)) :
            self.GetCombinationFeatureInOneMonth(A, B, month, ope)

    def GetFeature(self) :
        """
        feature list:
        1. gender of artist
        2. language of song
        3. how many days have been published for this song
        4. the number of total plays for current song in last whole month
        5. the number of total plays for current song in consecutive_recent days
        6. the number of total plays for current user and all the song in last whole month
        7. the number of total plays for current user and all the song in consecutive_recent days
        8. the proportion of the songs that the current user plays in last month
        9. the proportion of the songs that the currect user plays in consecutive_recent days
        10. the proportion of the artist that the current user plays in last month
        11. the proportion of the artist that the currect user plays in consecutive_recent days
        12. whether the current user have collected this song
        13. whether the current user have downloaded this song
        """
        logging.info("start extracting feature")

        self.SplitByMonth()

        self.GetSingleFeature('Gender', self.GetArtistGender)
        self.GetSingleFeature('Language', self.GetSongLanguage)
        self.GetSingleFeature('published_days', self.GetPublishedDays)
        self.GetSingleFeature('artist_id', self.GetArtistID)
        self.GetSingleFeature('song_init_plays', self.GetSongInitPlays)

        self.GetSingleFeature('total_plays_for_one_song_all', self.GetTotalPlaysForFeature)

# HourInterval
        HourInterval={'Morning':range(7,12),'Noon':range(12,15),'Afternoon':range(15,19),'Evening':range(19,25),'Midnight':range(1,7)}

        for when,interval in HourInterval.items():
            self.GetSingleFeature('total_plays_for_one_song_all_for_'+when, self.GetTotalPlaysForFeatureInHourInterval, condition_hour=interval)
        for consecutive_days in self.consecutive_recent_:
            self.GetSingleFeature('total_plays_for_one_song_recent_'+str(consecutive_days), self.GetTotalPlaysForFeature, consecutive_days=consecutive_days)
            for when,interval in HourInterval.items():
                self.GetSingleFeature('total_plays_for_one_song_recent_'+str(consecutive_days)+'_for_'+when, self.GetTotalPlaysForFeatureInHourInterval, consecutive_days=consecutive_days, condition_hour=interval)
        
#        for hour in xrange(24) :
#            self.GetSingleFeature('total_plays_for_one_song_all_for_hour_%d' % hour, self.GetTotalPlaysForFeatureInSpecificHour, condition_hour = hour)
#        for consecutive_days in self.consecutive_recent_:
#            self.GetSingleFeature('total_plays_for_one_song_recent_' + str(consecutive_days), self.GetTotalPlaysForFeature, consecutive_days = consecutive_days)
#            for hour in xrange(24) :
#                self.GetSingleFeature('total_plays_for_one_song_recent_' + str(consecutive_days) + '_for_hour_%d' % hour, self.GetTotalPlaysForFeatureInSpecificHour, consecutive_days = consecutive_days, condition_hour = hour)

        self.GetSingleFeature('is_collect', self.GetIsCollect)
        self.GetSingleFeature('is_download', self.GetIsDownload)
            
        self.GetTotalPlaysForArtist('total_plays_for_artist_all' ,'total_plays_for_one_song_all', 'artist_id')
        # get the plays of all artist, include the artist of the song
        self.GetTotalPlaysForEveryArtist('total_plays_for_every_artist_all','total_plays_for_one_song_all','artist_id')
        
        for when,interval in HourInterval.items():
            self.GetTotalPlaysForArtist('total_plays_for_artist_all_for_'+when, 'total_plays_for_one_song_all_for_'+when, 'artist_id')
        
        for consecutive_days in self.consecutive_recent_:
            self.GetTotalPlaysForArtist('total_plays_for_artist_recent_'+str(consecutive_days), 'total_plays_for_one_song_recent_' + str(consecutive_days), 'artist_id')
            for when,interval in HourInterval.items():
                self.GetTotalPlaysForArtist('total_plays_for_artist_recent_'+str(consecutive_days)+'_for_'+when, 'total_plays_for_one_song_recent_' + str(consecutive_days)+'_for_'+when, 'artist_id')

#        for hour in xrange(24) :
#            self.GetTotalPlaysForArtist('total_plays_for_artist_all_for_hour_%d' % hour ,'total_plays_for_one_song_all_for_hour_%d' % hour, 'artist_id')
#        for consecutive_days in self.consecutive_recent_:
#            self.GetTotalPlaysForArtist('total_plays_for_artist_recent_' + str(consecutive_days) ,'total_plays_for_one_song_recent_' + str(consecutive_days), 'artist_id')
#            for hour in xrange(24) :
#                self.GetTotalPlaysForArtist('total_plays_for_artist_recent_' + str(consecutive_days) + '_for_hour_%d' % hour ,'total_plays_for_one_song_recent_' + str(consecutive_days) + '_for_hour_%d' % hour, 'artist_id')

        self.GetCombinationFeature('total_plays_for_one_song_all', 'total_plays_for_artist_all', 'div') 
        for when,interval in HourInterval.items() :
            self.GetCombinationFeature('total_plays_for_artist_all_for_'+when, 'total_plays_for_artist_all', 'div') 
            self.GetCombinationFeature('total_plays_for_one_song_all_for_'+when, 'total_plays_for_artist_all_for_'+when, 'div') 
            self.GetCombinationFeature('total_plays_for_one_song_all_for_'+when, 'total_plays_for_one_song_all', 'div') 
            
        for consecutive_days in self.consecutive_recent_:
            self.GetCombinationFeature('total_plays_for_one_song_recent_' + str(consecutive_days) ,'total_plays_for_artist_recent_' + str(consecutive_days), 'div')
            for when,interval in HourInterval.items() :
                self.GetCombinationFeature('total_plays_for_artist_recent_' + str(consecutive_days) + '_for_' + when,'total_plays_for_artist_recent_' + str(consecutive_days) , 'div')
                self.GetCombinationFeature('total_plays_for_one_song_recent_' + str(consecutive_days) + '_for_' + when,'total_plays_for_artist_recent_' + str(consecutive_days) + '_for_' + when, 'div')
                self.GetCombinationFeature('total_plays_for_one_song_recent_' + str(consecutive_days) + '_for_' + when,'total_plays_for_one_song_recent_' + str(consecutive_days), 'div')
  
#        for hour in xrange(24) :
#            self.GetCombinationFeature('total_plays_for_artist_all_for_hour_%d' % hour, 'total_plays_for_artist_all', 'div') 
#            self.GetCombinationFeature('total_plays_for_one_song_all_for_hour_%d' % hour, 'total_plays_for_artist_all_for_hour_%d' % hour, 'div') 
#            self.GetCombinationFeature('total_plays_for_one_song_all_for_hour_%d' % hour, 'total_plays_for_one_song_all', 'div') 
            
#        for consecutive_days in self.consecutive_recent_:
#            self.GetCombinationFeature('total_plays_for_one_song_recent_' + str(consecutive_days) ,'total_plays_for_artist_recent_' + str(consecutive_days), 'div')
#            for hour in xrange(24) :
#                self.GetCombinationFeature('total_plays_for_artist_recent_' + str(consecutive_days) + '_for_hour_%d' % hour,'total_plays_for_artist_recent_' + str(consecutive_days) , 'div')
#                self.GetCombinationFeature('total_plays_for_one_song_recent_' + str(consecutive_days) + '_for_hour_%d' % hour,'total_plays_for_artist_recent_' + str(consecutive_days) + '_for_hour_%d' % hour, 'div')
#                self.GetCombinationFeature('total_plays_for_one_song_recent_' + str(consecutive_days) + '_for_hour_%d' % hour,'total_plays_for_one_song_recent_' + str(consecutive_days), 'div')
 
        self.GetLabel('label_plays', self.GetTotalPlaysForLabel)
        self.GetLabel('label_weekday', self.GetWeekday)
        self.GetLabel('label_day', self.GetDay)

        del self.data_
        del self.month_data_

def usage() :
    """
    """
    print 'feature_extract.py usage:'
    print '-h, --help: print help message'
    print '-j, --jobs: the number of processes to handler, default = 1'
    print '-t, --type: the type of data need to handler, default = unit'

if __name__ == '__main__' :
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hj:t:', ['type=', 'jobs=', 'help'])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    n_jobs = 1
    type = 'unit'
    for o, a in opts:
        if o in ('-h', '--help') :
            usage()
            sys.exit(1)
        elif o in ('-j', '--jobs') :
            print a
            n_jobs = int(a)
        elif o in ('-t', '--type') :
            type = a
        else:
            print 'invalid parameter:', o
            usage()
            sys.exit(1)

    feature_extract = FeatureExtract(type = type, n_jobs = n_jobs)

