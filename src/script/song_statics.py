#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

from configure import *
from utils import data_handler

def main (type = 'unit') :
   """
   """
   training, validation, testing = data_handler.GetData(type = type)
   print training.shape, validation.shape
   data = pd.concat([training, validation])
   data['label_day'] = data['label_day'].map(lambda x : '%02d' % int(x))
   data['datetime'] = data['month'].astype(str) + data['label_day']
   ds = sorted(list(set(data['datetime'].values.tolist())))
   print len(ds)
   artists = ['2b7fedeea967becd9408b896de8ff903', '8fb3cef29f2c266af4c9ecef3b780e97', '4ee3f9c90101073c99d5440b41f07daa', '6f462b173b2d6d20a2c9fb1ec0fd2dda']
   os.system('mkdir ' + ROOT + '/doc/source/song_statics/') 
   for artist in artists :
       filepath = ROOT + '/doc/source/song_statics/' + artist 
       os.system('mkdir ' + filepath)
       data_artist = data[data.artist_id == artist]
       groups = data_artist.groupby(['song_id']).groups
       for song, group in groups.items() :
            a = data_artist.loc[group][['datetime', 'label_plays']].values
            a = sorted(a, key = lambda v : v[0])
            x = range(1, len(ds) + 1)
            y = []
            i = 0
            j = 0
            while i < len(ds) :
                if j < len(a) and a[j][0] == ds[i] :
                    y.append(a[j][1])
                    j += 1
                else :
                    y.append(0)
                i += 1
            plt.plot(x, y)
            plt.plot(x, y, 'or')
            plt.savefig(filepath + '/' + song + '.jpg')
            plt.cla()
            plt.clf()
            plt.close()
            
       



if __name__ == '__main__' :
    main('full')

