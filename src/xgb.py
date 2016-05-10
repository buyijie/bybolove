#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import logging
import xgboost as xgb
import logging.config
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from utils import feature_reduction, evaluate
import sys
import getopt
import solver
from solver import HandlePredict

sys.path.insert(0, '..')
from configure import *

logging.config.fileConfig('logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))

def usage() :
    """
    """
    print 'xgboost.py usage:'
    print '-h, --help: print help message'
    print '-t, --type: the type of data need to handler, default = unit'

def xgboost_solver(train_x, train_y, validation_x, test_x, now_time , validation_y = np.array([]), feature_names = [], validation_artist_id=None, validation_month=None, validation_label_day=None):
    """
    """
    
    dtrain=xgb.DMatrix(train_x,label=train_y,feature_names=feature_names)
    dvalidation=xgb.DMatrix(validation_x,validation_y,feature_names=feature_names)
    dtest=xgb.DMatrix(test_x,feature_names=feature_names)

    logging.info('start training the xgboost model')
    params = {
        'eta' : 0.03,
        'silent': 1,
        'objective' : 'reg:linear',
        'max_depth' : 4,
        'seed' : 1000000007,
    }

    watchlist=[(dtrain,'train'),(dvalidation,'validation')]

    max_num_round=750
    best_num_round=0
    best_val=float('-Inf')
    curr_round=0
    curr_val=0
    history_train_val=[]
    history_validation_val=[]
    interval=10

    assert max_num_round%interval==0, "max_num_round must be multiple of interval"
    evals_result={}

    bst=None
    for step in xrange(max_num_round/interval):
        bst=xgb.train(params,dtrain,interval,watchlist,evals_result=evals_result,xgb_model=bst)
        curr_round+=10
        logging.info('current round is: %d' % curr_round)
        predict=bst.predict(dvalidation)
        predict=HandlePredict(predict.tolist())
        curr_val=evaluate.evaluate(predict, validation_y.tolist(), validation_artist_id, validation_month, validation_label_day)
        history_validation_val.append(curr_val)
        # train_val is rmse, not final score
        history_train_val.append(evals_result['train']['rmse'][-1])
        logging.info('the current score is %.10f' % curr_val)
        if curr_val > best_val:
            best_num_round=curr_round
            best_val=curr_val
            bst.save_model(ROOT+'/result/'+now_time+'/model/xgboost.model')


    bst=xgb.Booster(model_file=ROOT+'/result/'+now_time+'/model/xgboost.model')
    predict = bst.predict(dvalidation)

    with open(ROOT + '/result/' + now_time + '/parameters.param', 'w') as out :
        for key, val in params.items():
            out.write(str(key) + ': ' + str(val) + '\n')
        out.write('max_num_round: '+str(max_num_round)+'\n')
        out.write('best_num_round: '+str(best_num_round)+'\n')

    if validation_y.shape[0]  :
        logging.info('the loss in Training set is %.4f' % mean_squared_error(train_y, bst.predict(dtrain)))
        logging.info('the loss in Validation_set is %.4f' % mean_squared_error(validation_y, bst.predict(dvalidation)))

        plt.figure(figsize=(12, 6))
        # Plot feature importance
        plt.subplot(1, 2, 1)
        if (feature_names) == 0:
            feature_names = [str(i + 1) for i in xrange(validation_x.shape[0])]
        feature_names = np.array(feature_names)
        feature_importance_tmp = bst.get_fscore()
        feature_importance = np.array([ feature_importance_tmp[c] if c in feature_importance_tmp else 0 for c in feature_names ],dtype=np.float64)
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, feature_names[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')


        # Plot training deviance
        plt.subplot(1, 2, 2)
        validation_score = evals_result['validation']['rmse']
        train_score = evals_result['train']['rmse']
        plt.title('Deviance')
        plt.plot(np.arange(max_num_round/interval) + 1, history_train_val, 'b-',
                          label='Training Set Deviance')
        plt.plot(np.arange(max_num_round/interval) + 1, history_validation_val, 'r-',
                          label='Validation Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')

        plt.savefig(ROOT + '/result/' + now_time + '/statistics.jpg')

        print "not zero prediction : %d " % sum( [ i!=0 for i in predict.astype(int).tolist()] )
        print "total number of train data : %d" % train_y.shape[0]
        print "not zero label train data : %d" % sum(train_y!=0)
        print "total number of validation data : %d" % validation_y.shape[0]
        print "not zero label validation data : %d" % sum(validation_y!=0)

    return predict, bst.predict(dtest)

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

    solver.main(xgboost_solver, type = type, dimreduce_func = feature_reduction.undo) 
    solver.main(xgboost_solver, gap_month=2, type=type, dimreduce_func = feature_reduction.undo)
    evaluate.mergeoutput()
