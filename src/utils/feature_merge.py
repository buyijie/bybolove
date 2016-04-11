#!/usr/bin/env python
# coding=utf-8

import os
import sys
import pkl
import copy
import calendar 
import pandas as pd 
import logging 
import logging.config 
import datetime
import getopt 
from multiprocessing import Process 
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

class FeatureMerge :
    """
    """
    def __init__ (self, type = "unit", n_jobs = 1, consecutive_recent = [14, 7, 3], gap_month = 1, gap_day = 0) :
        self.consecutive_recent_ = consecutive_recent
        self.gap_month_ = gap_month
        self.gap_day_ = gap_day
        self.type_ = type
        self.n_jobs_ = n_jobs
        self.month_name_ = pkl.grab(ROOT + '/data/month_name.pkl')
        self.Read()
        self.GetData()

    def Read(self) :
        """
        read data from csv file
        """
        logging.info('read data from csv file, the type is %s'%self.type_)
        if self.type_ == "unit" :
            self.songs_ = pd.read_csv(ROOT + '/data/mars_tianchi_songs_tiny.csv', header = None)
        elif self.type_ == 'full' :
            self.songs_ = pd.read_csv(ROOT + '/data/mars_tianchi_songs.csv', header = None)
        else:
            logging.error('Invalid type of data set, please choose unit or full')
            exit (1)

        self.songs_.columns = ['song_id', 'artist_id', 'publish_time', 'song_init_plays', 'Language', 'Gender']

    def GetFromFile(self, month, feature_name) :
        """
        """
        logging.info('get feature %s in month %s from file' %(feature_name, self.month_name_[month]))
        filepath = ROOT + '/data/' + feature_name + '_' + self.month_name_[month] + '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '_' + str(self.gap_month_) + '.csv'
        if not os.path.exists(filepath) :
            logging.error(filepath + ' doesn\'t exists!')
            exit(1)

        data = pd.read_csv(filepath)
        return data

    def GetSongInfoForMergingData(self, song, feature) :
        """
        """
        return self.songs_.loc[self.songs_.song_id == song, feature].values.tolist()[0]

    def MergingDataInFeatureProcess(self, process_id, data, feature_list, filepath) :
        """
        """
        logging.info('process %s start!' % process_id)
        logging.info('the size of data for process %s is (%d %d)' % (process_id, data.shape[0], data.shape[1]))
        groups = data.groupby(['song_id']).groups
        cnt = 0
        for name, group in groups.items():
            cnt += 1
            if cnt % 20 == 0 :
                logging.info('now is the %dth group for process %s' % (cnt, process_id))
            for feature in feature_list :
                data.loc[group,feature] = self.GetSongInfoForMergingData(name, feature)
        
        for feature in feature_list :
            if data[data[feature].isnull()].shape[0] > 0 :
                logging.error('still has null value after filling for feature %s %s' % (feature, process_id))
        data.to_csv(filepath + '_' + process_id + '.csv', index = False)

    def MergingDataInLabelProcess(self, process_id, st, ed, weekday_1st, filepath) :
        """
        """
        logging.info('process %s start!' % process_id)
        for i in xrange(st, ed) :
            logging.info('now is the %dth day for process %s' % (i + 1, process_id))
            day = i + 1
            weekday = (weekday_1st + i) % 7

            table = copy.deepcopy(self.table_)

            table['label_weekday'] = weekday
            table['label_day'] = day

            table.to_csv(filepath + '_' + str(i + 1) + '.csv', index = False)

    def MergeData(self, x, y, y_month) :
        """
        merge the feature and label by song_id
        """
        data = pd.merge(x, y, how = 'outer', on = 'song_id')
        data.sort_values(['song_id', 'label_day'])

        # y exist, x not exitst
        feature_list = ['Gender', 'artist_id', 'Language']

        if len(feature_list) > 0 :
            # TO DO: how to file the null value more efficient
            
            logging.info('the size of data before filling the null value in feature is (%d %d)' % data.shape)
            data_null = data[data[feature_list[0]].isnull()]
            logging.info('the number of data have null value in feature is %d' % data_null.shape[0])
            data = data[data[feature_list[0]].notnull()]
            groups = data_null.groupby(['song_id']).groups
            groups_items = groups.items()
            total_items = len(groups_items)

            seperate = [0]
            for i in xrange(1, self.n_jobs_) :
                cnt = (total_items - seperate[-1] + self.n_jobs_ - i) / (self.n_jobs_ - i + 1)
                seperate.append(seperate[-1] + cnt)
            seperate.append(total_items)

            filepath = ROOT + '/data/merge_data_feature_' + self.month_name_[y_month]

            processes = []
            for i in xrange(self.n_jobs_) :
                index_list = []
                for key, val in groups_items[seperate[i]:seperate[i + 1]] :
                    index_list.extend(val)
            
                process = Process(target = self.MergingDataInFeatureProcess, args = (str(i + 1), data_null.loc[index_list,:], feature_list, filepath))
                process.start()
                processes.append(process)
            for process in processes:
                process.join()

            for i in xrange(self.n_jobs_) :
                temp = pd.read_csv(filepath + '_' + str(i + 1) + '.csv')
                data = pd.concat([data, temp])
                os.remove(filepath + '_' + str(i + 1) + '.csv')

            for feature in feature_list :
                if data[data[feature].isnull()].shape[0] > 0 :
                    logging.error('still has null value after filling for feature %s' % feature)
            logging.info('the size of data after filling the null value in feature is (%d %d)' % data.shape)

        # x exist, y not exist
        self.table_ = data[data.label_day.isnull()]
        data = data[data.label_day.isnull() == False]

        # how many days in y_month
        days = calendar.monthrange(int(self.month_name_[y_month][:4]), int(self.month_name_[y_month][4:]))[1]
        # the weekday for 1st
        weekday_1st = StrToDate(self.month_name_[y_month] + '01').weekday()

        seperate = [0]
        for i in xrange(1, self.n_jobs_) :
            cnt = (days - seperate[-1] + self.n_jobs_ - i) / (self.n_jobs_ - i + 1)
            seperate.append(seperate[-1] + cnt)
        seperate.append(days)
        filepath = ROOT + '/data/merge_data_label_' + self.month_name_[y_month]

        processes = []
        for i in xrange(self.n_jobs_) :
            process = Process(target = self.MergingDataInLabelProcess, args = (str(i + 1), seperate[i], seperate[i + 1], weekday_1st, filepath))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

        for i in xrange(days) :
            temp = pd.read_csv(filepath + '_' + str(i + 1) + '.csv')
            data = pd.concat([data, temp])
            os.remove(filepath + '_' + str(i + 1) + '.csv')

        return data

    def GetData(self, month_for_test = 2) :
        """
        """
        feature_list = ['artist_id', 'Gender', 'Language', 'published_days', 'total_plays_for_one_song_all', 'is_collect', 'is_download']
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
            self.final_data_.loc[self.final_data_[label].isnull(), label] = 0
        for feature in feature_list:
            self.final_data_.loc[self.final_data_[feature].isnull(), feature] = 0

        logging.info('the final data size is (%d %d)' % self.final_data_.shape)

        testing = sum(cnt_month[-month_for_test:])
        path = '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '_' + str(self.gap_month_) + '.csv'

        self.final_data_[:-testing].to_csv(ROOT + '/data/final_data_training' + path, index = False)
        self.final_data_[-testing:].to_csv(ROOT + '/data/final_data_testing' + path, index = False)

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

    feature_merge = FeatureMerge(type = type, n_jobs = n_jobs)

