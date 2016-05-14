#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
import logging.config
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from utils import feature_reduction, evaluate
import sys
import getopt
import solver
from solver import HandlePredict
import pandas as pd

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

class Config():
    train_length=30
    test_length=1
    task_num=10
    lr=0.01
    train_epoch=1000

class MTGP():
    """
    multi-task gaussian process regression
    """
    def add_placeholder(self):
        """
        add placeholder for data from outside
        train_x is N
        trian_y is M*N
        """
        self.train_x=tf.placeholder(tf.float32, shape=[None,1])
        self.train_y=tf.placeholder(tf.float32, shape=[None,1])
        self.test_x=tf.placeholder(tf.float32, shape=[None,1])

    def add_variable(self):
        """
        add trainable variables, i.e., L (a M*M lower triangle matrix), sigma_f (parameter covariance_function),
        l (parameter of covariance_function), D (a M*M diagonal matrix for white noise)  
        where M is the number of task, N is the number of train point in one task

        Then get K_c from L, K_c = L * L.T 
        get K_x from placeholder self.train_x
        """
        self.sigma_f=tf.Variable(1.0)
        self.l=tf.Variable(1.0)
        tmp_D=tf.Variable(tf.ones([self.config.task_num, self.config.task_num]))
        self.D=tf.mul(tmp_D*tmp_D, tf.constant(np.eye(self.config.task_num).astype(np.float32)))
        tmp_L=tf.Variable(tf.ones([self.config.task_num, self.config.task_num]))*0.01
        arr=np.zeros([self.config.task_num, self.config.task_num]).astype(np.float32)
        for i in xrange(self.config.task_num):
            for j in xrange(self.config.task_num):
                if j>i:
                    break
                arr[i,j]=1.0
        self.L=tf.mul(tmp_L, tf.constant(arr))
        self.K_c=tf.matmul(self.L, tf.transpose(self.L))
        self.K_x=self.get_covariance_x(self.train_x, self.train_x)

    def get_covariance_x(self, x, x_hat):
        """
        using squared exponential covariance function
        """
        x_hat=tf.reshape(x_hat, [1,-1])
        diff=x-x_hat
        return self.sigma_f*self.sigma_f*tf.exp(-1*diff*diff/(2*self.l*self.l))

    def add_loss(self):
        """
        add self.K which is the Kronecker product of self.K_c and self.K_x
        add self.sigma , which is the NM*NM covariance matrix of train_y
        get negative log likelihood of train_y, see paper nips
        """
        temp_K=tf.tile(tf.reshape(self.K_c, [self.config.task_num, 1, self.config.task_num, 1]), [1, self.config.train_length, 1, self.config.train_length]) \
                * tf.tile(tf.reshape(self.K_x, [1, self.config.train_length, 1, self.config.train_length]), [self.config.task_num, 1, self.config.task_num, 1])
        self.K=tf.reshape(temp_K, [self.config.train_length*self.config.task_num, self.config.train_length*self.config.task_num])

        temp_D_I=tf.tile(tf.reshape(self.D, [self.config.task_num, 1, self.config.task_num, 1]), [1, self.config.train_length, 1, self.config.train_length]) \
                * tf.tile(tf.reshape(tf.constant(np.eye(self.config.train_length).astype(np.float32)), [1, self.config.train_length, 1, self.config.train_length]),
                          [self.config.task_num, 1, self.config.task_num, 1])
        D_I=tf.reshape(temp_D_I, [self.config.train_length*self.config.task_num, self.config.train_length*self.config.task_num])

        self.sigma=self.K+D_I

        self.loss=0.5*tf.matmul(tf.matmul(tf.transpose(self.train_y), tf.matrix_inverse(self.sigma)), self.train_y) \
                +0.5*tf.log(tf.matrix_determinant(self.sigma))

        
    def add_train_op(self):
        """
        get train_op
        """
        self.train_op=tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)
    
    def add_predict_op(self):
        """
        get predict
        todo: get variance of predictions
        """
        tmp_K_star=self.get_covariance_x(self.test_x, self.train_x)
        self.K_star=tf.reshape(
            tf.tile(tf.reshape(self.K_c, [self.config.task_num, 1, self.config.task_num, 1]), [1, self.config.test_length, 1, self.config.train_length]) \
                * tf.tile(tf.reshape(tmp_K_star, [1, self.config.test_length, 1, self.config.train_length]), [self.config.task_num, 1, self.config.task_num, 1]), \
            [self.config.test_length*self.config.task_num, self.config.train_length*self.config.task_num])

        self.predict_op=tf.matmul(tf.matmul(self.K_star, tf.matrix_inverse(self.sigma)), self.train_y)


    
    def __init__(self, config):
        self.config=config
        self.add_placeholder()
        self.add_variable()
        self.add_loss()
        self.add_train_op()
        self.add_predict_op()

    def train(self, session, train_x, train_y):
        """
        train MTGP model
        """

        feed={
            self.train_x: train_x,
            self.train_y: train_y
        }
        loss, _ =session.run([self.loss, self.train_op], feed_dict=feed)
        print loss
        return loss
        
    def predict(self, session, train_x, train_y, test_x):
        """
        predict test_x
        """
        feed={
            self.train_x: train_x,
            self.train_y: train_y,
            self.test_x: test_x
        }
        test_y=session.run(self.predict_op, feed_dict=feed)
        print type(test_y)
        print test_y
        return test_y.reshape([-1, self.config.test_length]).T

if __name__ == "__main__":

    ts=pd.read_csv(ROOT+'/data/ts_artist_play_matrix.csv')           
    ts_matrix=ts.values.astype(np.float32)
    artist_list=ts.columns.values
    config=Config()
    train_x=(np.arange(config.train_length).astype(np.float32)+1).reshape([-1,1])
    train_y=ts_matrix[0:config.train_length,0:config.task_num].T.reshape([-1,1])
    test_x=(np.arange(config.test_length).astype(np.float32)+config.train_length+1).reshape([-1,1])
    test_y=ts_matrix[config.train_length:config.train_length+config.test_length,0:config.task_num].T.reshape([-1,1])
    model=MTGP(config)
    session=tf.Session()
    session.run(tf.initialize_all_variables())
    print session.run([model.sigma, tf.matrix_determinant(model.sigma)], feed_dict={model.train_x: train_x, model.train_y: train_y})
    for _ in xrange(config.train_epoch):
        print 'Epoch {}'.format(_)
        model.train(session, train_x, train_y)

    predict_y=model.predict(session, train_x, train_y, test_x)
    print test_y
#Unit Test
    sys.exit(0)
    #todo evaluate
    #evaluate(test_y, predict_y)

