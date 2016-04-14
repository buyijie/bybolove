#!/usr/bin/env python
# coding=utf-8

import logging 
import math

def evaluate(predict, label, artist, month, day) :
    """
    """
    if len(predict) != len(label):
        logging.error('the number of predict is not match with label')
        exit(1)

    artist_day_predict = {}
    artist_day_label = {}
    for row in xrange(len(predict)) :
        artist_day = artist[row] + '#' + str(month[row]) + '#' + str(day[row])
        artist_day_predict.setdefault(artist_day, 0)
        artist_day_label.setdefault(artist_day, 0)

        artist_day_predict[artist_day] += predict[row]
        artist_day_label[artist_day] += label[row]

    artist_all = {}
    artist_error = {}
    
    for key in artist_day_predict.keys() :
        artist = key.split('#')[0]
        error = ((artist_day_predict[key] - artist_day_label[key] + 1) / (artist_day_label[key] + 1)) ** 2
        artist_all.setdefault(artist, 0)
        artist_error.setdefault(artist, 0)

        artist_all[artist] += artist_day_label[key]
        artist_error[artist] += error

    score = 0 
    for artist in artist_all.keys():
        score += (1 - math.sqrt(artist_error[artist]/ 61)) * math.sqrt(artist_all[artist])
    return score
