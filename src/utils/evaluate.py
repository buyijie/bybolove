#!/usr/bin/env python
# coding=utf-8

import logging 
import math

def evaluate(predict, label) :
    """
    """
    if len(predict) != label.shape[0] :
        logging.error('the number of predict is not match with label')
        exit(1)
    artist_all = {}
    artist_error = {}
    for row in xrange(len(predict)) :
        error = ((predict[row] - label.label.values[row]) / label.label.values[row]) ** 2

        artist_all.setdefault(label.artist_id.values[row], 0)
        artist_error.setdefault(label.artist_id.values[row], 0)

        artist_all[label.artist_id.values[row]] += label.label.values[row]
        artist_error[label.artist_id.values[row]] += error

    score = 0 
    for artist in artist_all.keys():
        score += (1 - math.sqrt(artist_error[artist])) * math.sqrt(artist_all[artist])
    return score
