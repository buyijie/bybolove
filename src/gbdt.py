#!/usr/bin/env python
# coding=utf-8

import datetime
import logging
import utils.data_set as data_set
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import logging.config
import utils.evaluate as evaluate
import sys

sys.path.insert(0, '..')
from configure import *

logging.config.fileConfig('logging.conf')
logging.addLevelName(logging.WARNING, "\033[1;34m[%s]\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m[%s]\033[1;0m" % logging.getLevelName(logging.ERROR))


def gbdt_solver(dataset) :
    """
    """
    params = {
        'n_estimators': 100,
        'learning_rate': 0.03,
        'random_state': 1000000007,
        'max_depth': 2,
        'warm_start': True}
    logging.info("the parameters of gbdt is :")
    print params
    gb = GradientBoostingClassifier(**params)
    gb.fit(dataset.train_x_.values, dataset.train_y_.label.values)
    return gb, gb.predict(dataset.val_x_.values)


if __name__ == "__main__":
    now = datetime.datetime.now()
    now = datetime.datetime.strftime(now, '%Y%m%d-%H:%M:%S')
    dataset = data_set.DataSet()
    for consecutive in xrange (3 , 10) :
        dataset.GetTrainingSet(consecutive = consecutive)
        dataset.GetValidationSet()
        dataset.FeatureHandler()
        gbdt, predict = gbdt_solver(dataset)
        logging.info('the score of prediction %.2f' % evaluate.evaluate(predict, dataset.val_y_))
        joblib.dump(gbdt, ROOT + '/result/model/gbdt_' + now + '.pkl')
        
