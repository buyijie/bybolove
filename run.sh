#!/bin/bash

cd src/utils
python data_statistic.py -t full
python feature_extract.py -j 8 -t full 
python feature_merge.py -j 8 -t full
cd ..
python xgb.py -t full

