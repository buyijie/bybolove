#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import GradientBoostingRegressor
import logging.config
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
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
    print '-t, --type: the type of data need to handler, default = unit'

def gbdt_solver(train_x, train_y, test_x, now_time , test_y = np.array([]), feature_names = []):
    """
    """
    logging.info('start training the gbdt model')
    params = {
        'n_estimators': 10,
        'learning_rate': 0.03,
        'random_state': 1000000007,
        'max_depth': 3,
        'verbose' : 2
    }

    with open(ROOT + '/result/' + now_time + '/parameters.param', 'w') as out :
        for key, val in params.items():
            out.write(str(key) + ': ' + str(val) + '\n')

    gb = GradientBoostingRegressor(**params)
    gb.fit(train_x, train_y)
    joblib.dump(gb, ROOT + '/result/' + now_time + '.pkl')
    predict = gb.predict(test_x)

    if test_y.shape[0]  :
        logging.info('the mean_squared_error in Training set is %.4f' % mean_squared_error(train_y, gb.predict(train_x)))
        logging.info('the mean_squared_error in Testing set is %.4f' % mean_squared_error(test_y, gb.predict(test_x)))

        plt.figure(figsize=(12, 6))
        # Plot feature importance
        plt.subplot(1, 2, 1)
        if (feature_names) == 0:
            feature_names = [str(i + 1) for i in xrange(test_x.shape[0])]
        feature_names = np.array(feature_names)
        feature_importance = gb.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, feature_names[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')


        # Plot training deviance
        plt.subplot(1, 2, 2)
        test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
        for i, y_pred in enumerate(gb.staged_predict(test_x)):
            test_score[i] = gb.loss_(test_y, y_pred)
        plt.title('Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, gb.train_score_, 'b-',
                          label='Training Set Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                          label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')

        plt.savefig(ROOT + '/result/' + now_time + '/statistics.jpg')

    return predict


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
        elif o in ('-t', '--type') :
            type = a
        else:
            print 'invalid parameter:', o
            usage()
            sys.exit(1)

    solver.main(gbdt_solver, type = type) 
