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
        self.month_days_ = pkl.grab(ROOT + '/data/month_days.pkl')
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
        self.artist_list_ = sorted(set(self.songs_['artist_id']))
        self.songs_list_ = sorted(set(self.songs_['song_id']))

    def GetFromFile(self, month, feature_name) :
        """
        """
        if type(month) == int:
            month_name = self.month_name_[month]
        else :
            month_name = month
        logging.info('get feature %s in month %s from file' %(feature_name, month_name))
        filepath = ROOT + '/data/' + feature_name + '_' + month_name + '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '.csv'
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

    def MergeData(self, x, y, month_name) :
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

            filepath = ROOT + '/data/merge_data_feature_' + month_name

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

        # how many days in month_name 
        days = calendar.monthrange(int(month_name[:4]), int(month_name[4:]))[1]
        #days = self.month_days_[month]
        # the weekday for 1st
        weekday_1st = StrToDate(month_name + '01').weekday()

        seperate = [0]
        for i in xrange(1, self.n_jobs_) :
            cnt = (days - seperate[-1] + self.n_jobs_ - i) / (self.n_jobs_ - i + 1)
            seperate.append(seperate[-1] + cnt)
        seperate.append(days)
        filepath = ROOT + '/data/merge_data_label_' + month_name 

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

    def MergePlaysBetweenTwoMonth(self, x, y, x_days, y_days, month_name) :
        """
        """
        if x_days > y_days :
            x = x[x.label_day <= y_days]
        x_song = set(x.song_id.values.tolist())
        y_song = set(y.song_id.values.tolist())
        if x_days < y_days :
            tmp = []
            for song in x_song :
                avg = x[x.song_id == song].last_month_plays.mean()
                for day in xrange(x_days + 1 , y_days + 1) :
                    tmp.append([song, day, avg])
            tmp = pd.DataFrame(tmp, columns = x.columns)
            x = pd.concat([x, tmp])

        # x exist, y unexist
        tmp = []
        weekday_1st = StrToDate(month_name + '01').weekday()
        for song in (x_song - y_song) :
            for day in xrange(1, y_days + 1) :
                # song_id, label_plays, label_day, label_weekday
                tmp.append([song, 0, day, (weekday_1st + day - 1) % 7])
        tmp = pd.DataFrame(tmp, columns = y.columns)
        y = pd.concat([y, tmp])


        # y exist, x unexist
        tmp = []
        for song in (y_song - x_song) :
            for day in xrange(1, y_days + 1) :
                tmp.append([song, day, 0])
        tmp = pd.DataFrame(tmp, columns = x.columns)
        x = pd.concat([x, tmp])

        assert x.shape[0] == y.shape[0]
        
        data = pd.merge(x, y, how = 'inner', on = ['song_id', 'label_day']) 

        return data

    def GetDataForOneMonth(self, month) :
        """
        """        
        feature_list = ['artist_id', 'Gender', 'Language', 'published_days', 'total_plays_for_one_song_all', 'total_plays_for_artist_all','is_collect', 'is_download', 'song_init_plays']
        #for idx in xrange(len(self.artist_list_)):
        #    feature_list.append('total_plays_for_every_artist_all_'+str(idx))
        for idx in xrange(len(self.artist_list_)) :
            feature_list.append('CovarianceBetweenSongAndArtist_%d_mul_total_plays_for_every_artist_all_%d' % (idx, idx))
        
#        HourInterval={'Morning':range(7,12),'Noon':range(12,15),'Afternoon':range(15,19),'Evening':range(19,25),'Midnight':range(1,7)}
        
#        for when,interval in HourInterval.items() :
#            feature_list.append('total_plays_for_one_song_all_for_'+when)
#            feature_list.append('total_plays_for_one_song_all_for_'+when+'_div_total_plays_for_one_song_all')

        feature_list.append('median_plays_for_one_song_recent_30')
        for consecutive_days in self.consecutive_recent_ :
            feature_list.append('total_plays_for_one_song_recent_' + str(consecutive_days))
            feature_list.append('median_plays_for_one_song_recent_'+str(consecutive_days))
#            for when,interval in HourInterval.items() :
#                feature_list.append('total_plays_for_one_song_recent_' + str(consecutive_days) + '_for_' + when)
#                feature_list.append('total_plays_for_one_song_recent_'+str(consecutive_days)+'_for_'+when+'_div_total_plays_for_one_song_recent_'+str(consecutive_days))
#        for when,interval in HourInterval.items() :
#            feature_list.append('total_plays_for_artist_all_for_' + when)
#            feature_list.append('total_plays_for_artist_all_for_'+when+'_div_total_plays_for_artist_all')
#            feature_list.append('total_plays_for_one_song_all_for_'+when+'_div_total_plays_for_artist_all_for_'+when)
        for consecutive_days in self.consecutive_recent_ :
            feature_list.append('total_plays_for_artist_recent_' + str(consecutive_days))
#            for when,interval in HourInterval.items() :
#                feature_list.append('total_plays_for_artist_recent_'+str(consecutive_days)+'_for_'+when)
#                feature_list.append('total_plays_for_artist_recent_'+str(consecutive_days)+'_for_'+when+'_div_total_plays_for_artist_recent_'+str(consecutive_days))
#                feature_list.append('total_plays_for_one_song_recent_'+str(consecutive_days)+'_for_'+when+'_div_total_plays_for_artist_recent_'+str(consecutive_days)+'_for_'+when)     
#        for hour in xrange(24) :
#            feature_list.append('total_plays_for_one_song_all_for_hour_%d' % hour)
#            feature_list.append('total_plays_for_one_song_all_for_hour_%d_div_total_plays_for_one_song_all' % hour)
#        for consecutive_days in self.consecutive_recent_ :
#            feature_list.append('total_plays_for_one_song_recent_' + str(consecutive_days))
#            for hour in xrange(24) :
#                feature_list.append('total_plays_for_one_song_recent_' + str(consecutive_days) + '_for_hour_%d' % hour)
#                feature_list.append('total_plays_for_one_song_recent_%d_for_hour_%d_div_total_plays_for_one_song_recent_%d' % (consecutive_days, hour, consecutive_days))
#        for hour in xrange(24) :
#            feature_list.append('total_plays_for_artist_all_for_hour_%d' % hour)
#            feature_list.append('total_plays_for_artist_all_for_hour_%d_div_total_plays_for_artist_all' % hour)
#            feature_list.append('total_plays_for_one_song_all_for_hour_%d_div_total_plays_for_artist_all_for_hour_%d' % (hour, hour))
#        for consecutive_days in self.consecutive_recent_ :
#            feature_list.append('total_plays_for_artist_recent_' + str(consecutive_days))
#            for hour in xrange(24) :
#                feature_list.append('total_plays_for_artist_recent_%d_for_hour_%d' % (consecutive_days, hour))
#                feature_list.append('total_plays_for_artist_recent_%d_for_hour_%d_div_total_plays_for_artist_recent_%d' % (consecutive_days, hour, consecutive_days))
#                feature_list.append('total_plays_for_one_song_recent_%d_for_hour_%d_div_total_plays_for_artist_recent_%d_for_hour_%d' % (consecutive_days, hour, consecutive_days, hour))
        feature_list.append('total_plays_for_one_song_all_div_total_plays_for_artist_all')
        for consecutive_days in self.consecutive_recent_ :
            feature_list.append('total_plays_for_one_song_recent_%s_div_total_plays_for_artist_recent_%s' % (str(consecutive_days), str(consecutive_days)))
        label_list = ['label_plays', 'label_day', 'label_weekday']

        x_data = self.GetFromFile(month, 'label_day')
        x_data['last_month_plays'] = self.GetFromFile(month, 'label_plays').label_plays

        nxt_days = 0
        # training or validation
        if month + self.gap_month_ < len(self.month_name_) :
            this_month = self.month_name_[month + self.gap_month_]
            first = True
            y_data = None
            for label in label_list :
                data = self.GetFromFile(month + self.gap_month_, label)
                if first : y_data = data
                else : y_data[label] = data[label]
                first = False
            nxt_days = self.month_days_[month + self.gap_month_]
        # testing
        else :
            date = StrToDate(self.month_name_[month] + '01')
            new_month = date.month + self.gap_month_
            new_year = date.year
            new_day = date.day
            if new_month > 12 :
                new_year += 1
                new_month -= 12
            date = date.replace(year=new_year, month=new_month, day=new_day)
            this_month = DateToStr(date)[:6]

            nxt_days = calendar.monthrange(int(this_month[:4]), int(this_month[4:]))[1]
            y_data = []
            for song in self.songs_list_ : 
                for day in xrange(1, nxt_days + 1) :
                    y_data.append([song, 0, day, StrToDate(this_month + str(day)).weekday()])

            y_data = pd.DataFrame(y_data)
            y_data.columns = ['song_id', 'label_plays', 'label_day', 'label_weekday'] 

        y_data = self.MergePlaysBetweenTwoMonth(x_data, y_data, self.month_days_[month], nxt_days, this_month)
        first = True
        x_data = None 
        for feature in feature_list:
            data = self.GetFromFile(month, feature)
            if first : x_data = data
            else : x_data[feature] = data[feature]
            first = False

    
        final_data = self.MergeData(x_data, y_data, this_month)
        final_data['month'] = this_month
        return final_data


    def GetData(self) :
        """
        """
        self.final_data_ = None
        first_month = True
        cnt_month = []

        for month in xrange(len(self.month_name_) - self.gap_month_) :
            final_data = self.GetDataForOneMonth(month)
            cnt_month.append(final_data.shape[0])
            if first_month:
                self.final_data_ = final_data
            else :
                self.final_data_ = pd.concat([self.final_data_, final_data])
            first_month = False

        final_data = self.GetDataForOneMonth(len(self.month_name_) - 1)
        cnt_month.append(final_data.shape[0])
        self.final_data_ = pd.concat([self.final_data_, final_data])
        
        # replace the null value by zero
        for column in self.final_data_.columns:
            self.final_data_.loc[self.final_data_[column].isnull(), column] = 0

        logging.info('the final data size is (%d %d)' % self.final_data_.shape)

        training = sum(cnt_month[:-3])
        validation = cnt_month[-3]
        testing = cnt_month[-2]
        path = '_' + self.type_ + '_' + '_'.join(map(str, self.consecutive_recent_)) + '_' + str(self.gap_month_) + '.csv'

        self.final_data_[:training].to_csv(ROOT + '/data/final_data_training' + path, index = False)
        self.final_data_[training:training+validation].to_csv(ROOT + '/data/final_data_validation' + path, index = False)
        self.final_data_[training+validation:].to_csv(ROOT + '/data/final_data_testing' + path, index = False)

if __name__ == '__main__' :
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hj:t:', ['type=', 'jobs=', 'help'])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    n_jobs = 1
    data_type = 'unit'
    for o, a in opts:
        if o in ('-h', '--help') :
            usage()
            sys.exit(1)
        elif o in ('-j', '--jobs') :
            print a
            n_jobs = int(a)
        elif o in ('-t', '--type') :
            data_type = a
        else:
            print 'invalid parameter:', o
            usage()
            sys.exit(1)

    feature_merge = FeatureMerge(type = data_type, n_jobs = n_jobs)
    feature_merge = FeatureMerge(type = data_type, n_jobs = n_jobs, gap_month = 2)

