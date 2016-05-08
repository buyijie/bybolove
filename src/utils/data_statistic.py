#!/usr/bin/env python
# coding=utf-8

import sys
import time
import logging
import logging.config
import pandas as pd

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from configure import *

logging.config.fileConfig('../logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))

class DataStatistic :
    """
    """
    def __init__ (self, type = 'unit') :
        """
        """
        self.type_ = type
        self.Read()
        self.Join()
        self.GetMonth()
        self.GetHowManyDaysInEachMonth()
        self.GetAveragePlaysForEachSongInEachMonth()
        self.GetPlaysForEachSongInEachDay()
        self.GetAveragePlaysForEachArtistInEachMonth()
        self.GetPlaysForEachArtistInEachDay()

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

    def GetHowManyDaysInEachMonth(self) :
        """
        """
        days = list(set(self.data_.Ds.values.tolist()))
        self.days_in_month_ = {}
        for day in days :
            self.days_in_month_.setdefault(day[:6], 0)
            self.days_in_month_[day[:6]] += 1

    def GetMonth(self) :
        """
        """
        self.month_name_ = list(set([v[:6] for v in self.data_.Ds.values.tolist()]))
        self.month_name_.sort()

    def Join(self) :
        """
        """
        logging.info("merge action and songs by song_id")
        self.data_ = pd.merge(self.action_, self.songs_, how = 'left', on = 'song_id')
        self.data_['Ds'] = self.data_['Ds'].map(lambda v: str(v))
        self.data_['month'] = self.data_['Ds'].map(lambda v: str(v)[:6])
        self.data_['publish_time'] = self.data_['publish_time'].map(lambda v: str(v))
        self.data_['gmt_create'] = self.data_['gmt_create'].map(lambda v : time.gmtime(int(v)).tm_hour)
        logging.info('the size of whole data after joining is (%d %d)' % self.data_.shape)

    def GetPlaysForEachSongInEachDay(self, feature_name = 'GetPlaysForEachSongInEachDay') :
        """
        """
        logging.info('GetPlaysForEachSongInEachDay start!')
        filepath = ROOT + '/data/' + feature_name + '_' + self.type_ + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        average = pd.read_csv(ROOT + '/data/GetAveragePlaysForEachSongInEachMonth_' + self.type_ + '.csv')
        data_plays = self.data_[self.data_.action_type == 1]
        groups = data_plays.groupby(['song_id', 'Ds']).groups
        with open(filepath, 'w') as out :
            out.write('song_id,month,Ds,plays\n')
            for key, values in groups.items() :
                song_id, Ds = key
                out.write('%s,%s,%s,%.10f\n' % (song_id, Ds[:6], Ds, len(values) - average[(average.month == int(Ds[:6])) & (average.song_id == song_id)].plays.values))

    def GetAveragePlaysForEachSongInEachMonth(self, feature_name = 'GetAveragePlaysForEachSongInEachMonth') :
        """
        """
        logging.info('GetAveragePlaysForEachSongInEachMonth start!')
        filepath = ROOT + '/data/' + feature_name + '_' + self.type_ + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        data_plays = self.data_[self.data_.action_type == 1] 
        groups = data_plays.groupby(['song_id', 'month']).groups
        with open(filepath, 'w') as out :
            out.write('song_id,month,plays\n')
            for key, values in groups.items() :
                song_id, month= key
                out.write('%s,%s,%.10f\n' % (song_id, month, len(values) * 1.0 / self.days_in_month_.get(month, 0)))

    def GetPlaysForEachArtistInEachDay(self, feature_name = 'GetPlaysForEachArtistInEachDay') :
        """
        """
        logging.info('GetPlaysForEachArtistInEachDay start!')
        filepath = ROOT + '/data/' + feature_name + '_' + self.type_ + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        average = pd.read_csv(ROOT + '/data/GetAveragePlaysForEachArtistInEachMonth_' + self.type_ + '.csv')
        data_plays = self.data_[self.data_.action_type == 1] 
        groups = data_plays.groupby(['artist_id', 'Ds']).groups
        with open(filepath, 'w') as out :
            out.write('artist_id,month,Ds,plays\n')
            for key, values in groups.items() :
                artist_id, Ds = key
                out.write('%s,%s,%s,%.10f\n' % (artist_id, Ds[:6], Ds, len(values) - average[(average.month == int(Ds[:6])) & (average.artist_id == artist_id)].plays.values))

    def GetAveragePlaysForEachArtistInEachMonth(self, feature_name = 'GetAveragePlaysForEachArtistInEachMonth') :
        """
        """
        logging.info('GetAveragePlaysForEachArtistInEachMonth start!')
        filepath = ROOT + '/data/' + feature_name + '_' + self.type_ + '.csv'
        if os.path.exists(filepath) :
            logging.info(filepath + ' exists!')
            return
        data_plays = self.data_[self.data_.action_type == 1] 
        groups = data_plays.groupby(['artist_id', 'month']).groups
        with open(filepath, 'w') as out :
            out.write('artist_id,month,plays\n')
            for key, values in groups.items() :
                artist_id, month = key
                out.write('%s,%s,%.10f\n' % (artist_id, month, len(values) * 1.0 / self.days_in_month_.get(month, 0)))


if __name__ == '__main__' :
    ds = DataStatistic()
