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
from utils.feature_handler import *
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

def xgboost_solver(train_x, train_y, validation_x, test_x, filepath, validation_y = np.array([]), feature_names = [], validation_artist_id=None, validation_month=None, validation_label_day=None, transform_type=0, shuffle=0):
    """
    transform_type: 0: no transform, 1: ratiolize predict, 2: loglize predict
    shuffle: every shuffle number, shuffle the train data
    """
#normalize feature
#    all_x=np.vstack([train_x, validation_x, test_x])
#    mean_x=np.mean(all_x, axis=0)
#    std_x=np.std(all_x, axis=0)
#    _col=std_x>0
#    del all_x
#    train_x[:, _col]=(train_x[:, _col]-mean_x[_col])/std_x[_col]
#    validation_x[:, _col]=(validation_x[:, _col]-mean_x[_col])/std_x[_col]
#    test_x[:, _col]=(test_x[:, _col]-mean_x[_col])/std_x[_col]
    
#add validation data to train data
#    train_x=np.vstack([train_x, validation_x])
#    train_y=np.hstack([train_y, validation_y])

#    col_last_month_plays=None
#    for i in xrange(len(feature_names)):
#        if feature_names[i]=='last_month_plays':
#            col_last_month_plays=i
#    assert col_last_month_plays is not None, 'No feature last_month_plays found!'
#
#    train_last_month_plays=train_x[:, col_last_month_plays]
#    validation_last_month_plays=validation_x[:, col_last_month_plays]
#    test_last_month_plays=test_x[:, col_last_month_plays]

#Transform predict
    dtrain=xgb.DMatrix(train_x, label=Transform(train_y, transform_type), feature_names=feature_names)
    dvalidation=xgb.DMatrix(validation_x, Transform(validation_y, transform_type), feature_names=feature_names)
    dtest=xgb.DMatrix(test_x, feature_names=feature_names)

    logging.info('start training the xgboost model')
    params = {
        'eta' : 0.03,
        'silent': 1,
        'objective' : 'reg:linear',
        'max_depth' : 7,
        'seed' : 1000000007,
        'gamma': 0,  # default 0, minimum loss reduction required to partition
        'min_child_weight':1000, # default 1, minimun number of instances in each node
        'alpha':0, # default 0, L1 norm
        'lambda':1, # default 1, L2 norm
    }

    watchlist=[(dtrain,'train'),(dvalidation,'validation')]

    max_num_round=300
    best_num_round=0
    best_val=float('-Inf')
    curr_round=0
    curr_val=0
    history_train_val=[]
    history_validation_val=[]
    interval=5

    assert max_num_round%interval==0, "max_num_round must be multiple of interval"
    evals_result={}

    bst=None
    for step in xrange(max_num_round/interval):
        bst=xgb.train(params,dtrain,interval,watchlist,evals_result=evals_result,xgb_model=bst)
        curr_round+=interval
        logging.info('current round is: %d' % curr_round)
        predict=bst.predict(dvalidation)
#detransform to plays
        predict=Convert2Plays(predict, transform_type)
        predict=HandlePredict(predict.tolist())
        curr_val=evaluate.evaluate(predict, validation_y.tolist(), validation_artist_id, validation_month, validation_label_day)
        history_validation_val.append(curr_val)
        # train_val is rmse, not final score
        history_train_val.append(evals_result['train']['rmse'][-1])
        logging.info('the current score is %.10f' % curr_val)
        if curr_val > best_val:
            best_num_round=curr_round
            best_val=curr_val
            bst.save_model(filepath +'/model/xgboost.model')
        #shuffle train_x train_y last_month_plays
        if (shuffle>0) and (curr_round%shuffle==0):
            indices=np.random.permutation(train_x.shape[0])
            dtrain=xgb.DMatrix(train_x[indices,:], label=Transform(train_y[indices], transform_type), feature_names=feature_names)
            logging.info('shuffle')

    dtrain=xgb.DMatrix(train_x, label=Transform(train_y, transform_type), feature_names=feature_names)
    bst=xgb.Booster(model_file=filepath +'/model/xgboost.model')
    predict = bst.predict(dvalidation)
#detransform to plays
    predict=Convert2Plays(predict, transform_type)

    with open(filepath + '/parameters.param', 'w') as out :
        for key, val in params.items():
            out.write(str(key) + ': ' + str(val) + '\n')
        out.write('max_num_round: '+str(max_num_round)+'\n')
        out.write('best_num_round: '+str(best_num_round)+'\n')
        out.write('transform_type: '+str(transform_type)+'\n')

    if validation_y.shape[0]  :
        logging.info('the loss in Training set is %.4f' % mean_squared_error(train_y, Convert2Plays(bst.predict(dtrain), transform_type)))
        logging.info('the loss in Validation_set is %.4f' % mean_squared_error(validation_y, Convert2Plays(bst.predict(dvalidation), transform_type)))

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

        plt.savefig(filepath + '/statistics.jpg')

        print "not zero prediction : %d " % sum( [ i!=0 for i in Convert2Plays(predict, transform_type).astype(int).tolist()] )
        print "total number of train data : %d" % train_y.shape[0]
        print "not zero label train data : %d" % sum(train_y!=0)
        print "total number of validation data : %d" % validation_y.shape[0]
        print "not zero label validation data : %d" % sum(validation_y!=0)
        print "best_num_round : %d" % best_num_round

    return predict, Convert2Plays(bst.predict(dtest), transform_type)

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hj:t:', ['type=', 'jobs=', 'help'])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    n_jobs = 1
    _type = 'unit'
    for o, a in opts:
        if o in ('-h', '--help') :
            usage()
            sys.exit(1)
        elif o in ('-t', '--type') :
            _type = a
        else:
            print 'invalid parameter:', o
            usage()
            sys.exit(1)
    
    solver.run(xgboost_solver, type=_type)
    #solver.main(xgboost_solver, type = _type, dimreduce_func = feature_reduction.undo, transform_type=0) 
    #solver.main(xgboost_solver, gap_month=2, type=_type, dimreduce_func = feature_reduction.undo, transform_type=2)
    #evaluate.mergeoutput()
