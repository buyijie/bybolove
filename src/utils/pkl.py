#!/usr/bin/env python
# coding=utf-8

import pickle

def store (input, filename) :
    """
    store input into filename used pickle.dump
    """
    cout = open (filename, 'w')
    pickle.dump (input, cout)
    cout.close ()

def grab (filename) :
    """
    load data from filename used pickle.load
    """
    cin = open (filename, 'r')
    return pickle.load (cin)
