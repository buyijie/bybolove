#!/usr/bin/env python
# coding=utf-8

import numpy as np
import time
import pandas as pd
import logging
import logging.config
import datetime
import getopt
from sklearn import linear_model

import sys
sys.path.insert(0, '..')
from configure import *

logging.config.fileConfig('logging.conf')
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

class Solver:
    """
    """
    def __init__ (self, type = "unit", n_jobs = 1, consecutive_recent = [14, 7, 3],  gap_day = 0) :
        self.consecutive_recent_ = consecutive_recent
        self.gap_day_ = gap_day
        self.type_ = type
        self.n_jobs_ = n_jobs
        self.Read()
        self.Join()
        self.main()

    def Read(self) :
        """
        read data from csv file
        """
        if self.type_ == "unit" :
            self.songs_ = pd.read_csv(ROOT + '/data/mars_tianchi_songs_tiny.csv', header = None)
            self.action_ = pd.read_csv(ROOT + '/data/mars_tianchi_user_actions_tiny.csv', header = None)
#            self.label_ = pd.read_csv(ROOT + '/data/label_tiny.csv', header = None)
        elif self.type_ == 'full' :
            self.songs_ = pd.read_csv(ROOT + '/data/mars_tianchi_songs.csv', header = None)
            self.action_ = pd.read_csv(ROOT + '/data/mars_tianchi_user_actions.csv', header = None)
#            self.label_ = pd.read_csv(ROOT + '/data/label.csv', header = None)
        else:
            logging.error('Invalid type of data set, please choose unit or full')
            exit (1)

        self.songs_.columns = ['song_id', 'artist_id', 'publish_time', 'song_init_plays', 'Language', 'Gender']
        self.action_.columns = ['user_id', 'song_id', 'gmt_create', 'action_type', 'Ds']
#        self.label_.columns = ['artist_id', 'Plays', 'Ds']
        self.song_id_set_ = self.songs_.song_id.values.tolist().sort()
        self.artist_list_ = sorted(set(self.songs_['artist_id'])) 

    def Join(self) :
        """
        """
        self.data_ = pd.merge(self.action_, self.songs_, how = 'left', on = 'song_id')
        self.data_['Ds'] = self.data_['Ds'].map(lambda v: str(v))
        self.data_['publish_time'] = self.data_['publish_time'].map(lambda v: str(v))
        self.data_['gmt_create'] = self.data_['gmt_create'].map(lambda v : time.gmtime(int(v)).tm_hour)
        del self.action_
    
    def GetData(self, artist_data, days = 10) :
        usingdate = [] 
        for day in xrange(31 - days, 31) :
            usingdate.append('201508%02d' % day)
        using_data = []
        for date in usingdate :
            using_data.append(artist_data[(artist_data.Ds == date) & (artist_data.action_type == 1)].shape[0])
        return using_data

    def ExtractMedium(self, artist_data, days = 10, remove_large = 4, remove_small = 4) :
        using_data = self.GetData(artist_data, days = days)
        index = np.argsort(using_data)
        using_data = [using_data[tp] for tp in sorted(index[remove_small:-remove_large])]
        return using_data

    def Regression(self, artist_data, days = 30, remove_large = 12, remove_small = 2, target = 3) :
        using_data = self.ExtractMedium(artist_data, days = days, remove_small = remove_small, remove_large = remove_large)
        regr = linear_model.LinearRegression()
        train_x = np.array([[i] for i in xrange(len(using_data))]).reshape(-1, 1)
        train_y = np.array(using_data).reshape(-1, 1)
        regr.fit(train_x, train_y)

        predict = []
        for i in xrange(target) :
            predict.append(int(regr.predict(np.array([i + len(using_data)]).reshape(-1, 1))[0][0]))
        #print using_data, predict
        return predict
    
    def main (self) :
        artist_group = self.data_.groupby(['artist_id']).groups
        
        targetdate = []
        for month in xrange(9, 11) :
            for day in xrange(1, 31) :
                targetdate.append('2015%02d%02d' % (month, day))
        for artist, group in artist_group.items() :
            artist_data = self.data_.loc[group]
            using_data = self.ExtractMedium(artist_data)
            #using_data = self.Regression(artist_data)

            for i in xrange(len(targetdate)) :
                print artist + ',' + str(using_data[i % len(using_data)]) + ',' + targetdate[i] 


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
    solver = Solver(type = type, n_jobs = n_jobs)

