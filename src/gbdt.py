#!/usr/bin/env python
# coding=utf-8

import datetime
import logging
import utils.data_set as data_set
from sklearn.ensemble import GradientBoostingRegressor
import logging.config
import utils.evaluate as evaluate
from sklearn.externals import joblib
import sys
import getopt
import solver

sys.path.insert(0, '..')
from configure import *

logging.config.fileConfig('logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))

def usage() :
    """
    """
    print 'gbdt.py usage:'
    print '-h, --help: print help message'
    print '-j, --jobs: the number of processes to handler, default = 1'
    print '-t, --type: the type of data need to handler, default = unit'

def gbdt_solver(train_x, train_y, test_x, now_time) :
    """
    """
    logging.info('start training the gbdt model')
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'random_state': 1000000007,
        'max_depth': 3,
        'verbose' : 1
    }

    with open(ROOT + '/result/' + now_time + '.param', 'w') as out :
        for key, val in params.items():
            out.write(str(key) + ': ' + str(val) + '\n')

    gb = GradientBoostingRegressor(**params)
    gb.fit(train_x, train_y)
    joblib.dump(gb, ROOT + '/result/' + now_time + '.pkl')
    
    return gb.predict(test_x)


if __name__ == "__main__":
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

    dataset = data_set.DataSet(type = type, n_jobs = n_jobs)
    solver.main(dataset, gbdt_solver)
    

 
