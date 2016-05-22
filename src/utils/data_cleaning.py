#!/usr/bin/env python
# coding=utf-8

import os
import sys
import pkl
import getopt
import logging
import logging.config
import pandas as pd

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from configure import *

logging.config.fileConfig('../logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))

def read (type, gap_month = 1, consecutive_recent = [14, 7, 3]) : 
    """
    """
    path = '_' + type + '_' + '_'.join(map(str, consecutive_recent)) + '_' + str(gap_month) + '.csv'
    testing_path = ROOT + '/data/final_data_testing' + path
    validation_path = ROOT + '/data/final_data_validation' + path
    training_path = ROOT + '/data/final_data_training' + path
    if not os.path.exists(testing_path) :
        logging.error(testing_path + ' doesn\'t exists!')
        exit(1)
    if not os.path.exists(training_path) :
        logging.error(training_path + ' doesn\'t exists!')
        exit(1)
    if not os.path.exists(validation_path) :
        logging.error(validation_path+ ' doesn\'t exists!')
        exit(1)
    testing = pd.read_csv(testing_path)
    validation = pd.read_csv(validation_path)
    training = pd.read_csv(training_path)
    return training, validation, testing

def Filter(data) :
    """
    """
    # TO DO: 
    return data

def SongPlaysFilter(data, type = 'unit', gap_month = 1) :
    """
    """
    song_month_groups = data.groupby(['song_id', 'month']).groups
    for song_month, song_month_group in song_month_groups.items() :
        data.loc[song_month_group].label_plays = Filter(data.loc[song_month_group].label_plays.values)
    return data

def DiscardBySongPlays(data, type = 'unit', gap_month = 1) :
    """
    """
    filepath = ROOT + '/data/total_song_plays_' + type + '_' + str(gap_month)
    with open(filepath, 'w') as out :
        out.write('artist_id,song_id,total_plays,percentage\n')
    artist_groups = data.groupby(['artist_id']).groups
    song_removed = []
    song_retain = []
    for artist, artist_group in artist_groups.items() :
        data_artist = data.loc[artist_group]
        song_groups = data_artist.groupby(['song_id']).groups
        song_plays = []
        for song, song_group in song_groups.items() :
            song_plays.append([song, data_artist.loc[song_group].label_plays.sum()])
        song_plays.sort(key = lambda v : v[-1], reverse = True)
        allPlays = sum([v[-1] for v in song_plays])
        with open(filepath, 'a') as out :
            tot = 0
            remove_cnt = 0
            for song, val in song_plays :
                tot += val
                out.write('%s,%s,%.1f,%.5f\n' % (artist, song, val, tot * 1.0 / allPlays))
                if tot <= allPlays * 0.95 :
                    song_retain.append(song)
                else :
                    song_removed.append(song)
                    remove_cnt += 1
            logging.info('for artist %s, %d songs removed' % (artist, remove_cnt))
            out.write('#######################\n')

    filepath = ROOT + '/data/which_song_removed_' + type + '_' + str(gap_month)
    with open (filepath, 'w') as out :
        for song in song_removed:
            out.write(song + '\n')
    return data[data.song_id.isin(song_retain)]
    
def output (training, validation, testing, type = 'unit', gap_month = 1, consecutive_recent = [14, 7, 3]) :
    """
    """
    path = '_cleaning_' + type + '_' + '_'.join(map(str, consecutive_recent)) + '_' + str(gap_month) + '.csv'

    training.to_csv(ROOT + '/data/final_data_training' + path, index = False)
    validation.to_csv(ROOT + '/data/final_data_validation' + path, index = False)
    testing.to_csv(ROOT + '/data/final_data_testing' + path, index = False)

def main (type = 'unit', gap_month = 1) : 
    """
    """
    logging.info('do the data cleaning for type: %s, gap_month: %d' % (type, gap_month))
    training, validation, testing = read (type, gap_month = gap_month)
    training = SongPlaysFilter(training, type = type, gap_month = gap_month)
    training = DiscardBySongPlays(training, type = type, gap_month = gap_month)

    output (training, validation, testing, type = type, gap_month = gap_month)

def usage() :
    """
    """
    print 'data_cleaning.py usage:'
    print '-h, --help: print help message'
    print '-t, --type: the type of data need to handler, default = unit'

if __name__ == '__main__' :
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hj:t:', ['type=', 'jobs=', 'help'])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    type = 'unit'
    for o, a in opts:
        if o in ('-h', '--help') :
            usage()
            sys.exit(1)
        elif o in ('-t', '--type') :
            type = a
        else:
            print 'invalid parameter:', o
            usage()
            sys.exit(1)
    main(type, gap_month = 1)
    main(type, gap_month = 2)
