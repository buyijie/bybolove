#!/usr/bin/env python
# coding=utf-8

import os
import math
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
    def __init__ (self, type = "unit", n_jobs = 1, consecutive_recent = [14, 7, 3], gap_month = 1, gap_day = 0) :
        self.consecutive_recent_ = consecutive_recent
        self.gap_month_ = gap_month
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

    def Join(self) :
        """
        """
        logging.info("merge action and songs by song_id")
        self.data_ = pd.merge(self.action_, self.songs_, how = 'left', on = 'song_id')
        self.data_['Ds'] = self.data_['Ds'].map(lambda v: str(v))
        self.data_['publish_time'] = self.data_['publish_time'].map(lambda v: str(v))
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
        self.month_data_[-1] = self.month_data_[-1].sort_values(['user_id', 'song_id', 'Ds'], ascending = True)
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
        return len(sub.groupby(['action_type']).groups.get(1, []))

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

    def SingleFeatureProcess(self, id, month, function, st, ed, file_path, consecutive_days = None):
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
                value = function(sub)
            else :
                lastday = calendar.monthrange(int(self.month_name_[month][:4]), int(self.month_name_[month][4:]))[1]
                begin_day = StrToDate(self.month_name_[month][:4] + str(lastday - consecutive_days + 1))

                song_date_st = 0
                while song_date_st < n and StrToDate(sub.iloc[song_date_st,:].Ds) < begin_day :
                    song_date_st += 1
                value = function(sub[song_date_st:])

            if value != None:
                with open(file_path, 'a') as out :
                    out.write(entry.song_id + ',' + str(value) + '\n')

            song_st = song_ed

    def LabelProcess(self, id, month, function, st, ed, file_path, consecutive_days = None):
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

            song_date_st = 0
            begin_day = StrToDate(self.month_name_[month] + '01')
            today = begin_day
            while today.month == begin_day.month:
                if song_date_st < n and DateToStr(today) == sub.iloc[song_date_st,:].Ds :
                    song_date_ed = self.GetSameSongDatePair(month, song_date_st)
                    value = function(month, today, sub.iloc[song_date_st:song_date_ed])
                    song_date_st = song_date_ed
                else:
                    value = function(month, today, sub.iloc[0:0])

                if value != None:
                    with open(file_path, 'a') as out :
                        out.write(entry.song_id + ',' + str(value) + '\n')
                today = today + one_day

            song_st = song_ed

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
            point = self.GetSameSong(month, point)
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
            out.write ('song_id,' + feature_name + '\n')
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

        self.GetSingleFeature('total_plays_for_one_song_all', self.GetTotalPlaysForFeature)
        for consecutive_days in self.consecutive_recent_:
            self.GetSingleFeature('total_plays_for_one_song_recent_' + str(consecutive_days), self.GetTotalPlaysForFeature, consecutive_days)

        self.GetSingleFeature('is_collect', self.GetIsCollect)
        self.GetSingleFeature('is_download', self.GetIsDownload)

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

