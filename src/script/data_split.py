#!/usr/bin/env python
# coding=utf-8

import os
import sys
import datetime
import logging
import logging.config

logging.config.fileConfig('../logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))

songs_file = '../../data/mars_tianchi_songs.csv'
action_file = '../../data/mars_tianchi_user_actions.csv'
action_training_file = '../../data/mars_tianchi_user_actions_training.csv'
action_testing_file = '../../data/mars_tianchi_user_actions_testing.csv'
songs_tiny_file = '../../data/mars_tianchi_songs_tiny.csv'
action_tiny_file = '../../data/mars_tianchi_user_actions_tiny.csv'
action_tiny_testing_file = '../../data/mars_tianchi_user_actions_tiny_testing.csv'
action_tiny_training_file = '../../data/mars_tianchi_user_actions_tiny_training.csv'
label_file = '../../data/label.csv'
label_tiny_file = '../../data/label_tiny.csv'
songs_singers = {}

def CheckAndDelete(file) :
    """
    if the file exists, remove it
    parameters:
        @file: the path of file which need to check
    """
    if os.path.exists(file) :
        os.remove(file)

def TrainTestSplit(action_file, days, training_file = action_training_file, testing_file = action_testing_file) :
    """
    split training set and testing set for training process
    parameters:
        @action_file: the file of action data 
        @days: how many days of data is used for testing 
    """
    logging.info('Begin: TrainTestSplit')
    last_day = datetime.datetime.strptime('20150830', '%Y%m%d')
    total , training, testing = 0, 0, 0
    CheckAndDelete(training_file)
    CheckAndDelete(testing_file)
    action_testing_io = open(testing_file, 'a')
    action_training_io = open(training_file, 'a')
    action = open(action_file, 'r')
    for each in action:
        total += 1
        if total % 100000 == 0:
            logging.info('%d-th line of data' % total)
        this_day = datetime.datetime.strptime(each.strip().split(',')[4], '%Y%m%d')
        if (last_day - this_day).days < days:
            testing += 1
            action_testing_io.write(each)
        else:
            training += 1
            action_training_io.write(each)
    logging.info('TrainTestSplit Done! total: %d training: %d testing: %d' % (total, training, testing))
    action.close ()
    action_testing_io.close ()
    action_training_io.close ()

def ExtractTinyDataForUnitTest(songs_file, action_file, days, singer_be_chosen = 2) :
    """
    extract some tiny data for unit testing
    parameters:
        @songs_file: the file of singers and songs data
        @action_file: the file of action data
        @days: used in TrainTestSplit
        @singer_be_chosen: hom many singers is used for tiny data
    """
    songs = open(songs_file, 'r')
    action = open(action_file, 'r')
    CheckAndDelete(action_tiny_file)
    CheckAndDelete(songs_tiny_file)
    action_tiny_io = open(action_tiny_file, 'a') 
    songs_tiny_io = open(songs_tiny_file, 'a')
    singers_set = set ()
    songs_set = set ()
    action_cnt = 0
    songs_cnt = 0
    for each in songs :
        line = each.split (',')
        songs_singers[line[0]] = line[1]
        if line[1] not in singers_set:
            if len(singers_set) >= singer_be_chosen : continue
        
        singers_set.add(line[1])
        songs_set.add(line[0])
        songs_cnt += 1
        songs_tiny_io.write(each)

    for each in action:
        line = each.split (',')
        if line[1] in songs_set:
            action_tiny_io.write(each)
            action_cnt += 1

    logging.info("ExtractTinyDataForUnitTest Done! %d singers, %d songs, %d action." % (singer_be_chosen, songs_cnt, action_cnt))

    action_tiny_io.close ()
    songs_tiny_io.close ()
    TrainTestSplit(action_tiny_file, days, training_file = action_tiny_training_file, testing_file = action_tiny_testing_file)

def GenerateLabelFile(testing_file, target_file) :
    """
    generate label file for final evaluation
    <singers>,<count>,<date>
    parameters:
        testing_file: the testing file need to handle
        target_file: the target file is wrote into
    """
    CheckAndDelete(target_file)
    data = open(testing_file, 'r')
    target_file_io = open(target_file, 'a')
    mymap = {}
    for each in data :
        line = each.strip().split(',')
        # only the play operation is considerable
        if line[3] != '1' : continue

        key = songs_singers[line[1]] + line[4]
        mymap.setdefault(key, 0)
        mymap[key] += 1

    for key, value in mymap.items() :
        line = key[:-8] + ',' + str (value) + ',' + key[-8:]
        target_file_io.write(line + '\n')

    logging.info('GenerateLabelFile for %s done!' % testing_file)
        
if __name__ == '__main__' :
    if len(sys.argv) != 2:
        logging.error('Invalid number of parameters.\nUsage: python data_split.py <number of days for testing>')
        exit()  
    TrainTestSplit(action_file, int(sys.argv[1]))
    ExtractTinyDataForUnitTest(songs_file, action_file, int(sys.argv[1]))
    GenerateLabelFile(action_testing_file, label_file)
    GenerateLabelFile(action_tiny_testing_file, label_tiny_file)
    
