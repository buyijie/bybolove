#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
import logging.config
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from utils import feature_reduction, evaluate, feature_handler
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
    print 'nn.py usage:'
    print '-h, --help: print help message'
    print '-t, --type: the type of data need to handler, default = unit'

def xavier_weight_init():
    def _xavier_initializer(shape, **kwargs):
        _tmp=np.sqrt(6.0/(sum(shape)*1.0))
        out=tf.random_uniform(shape,-_tmp,_tmp)
        return out
    return _xavier_initializer

def data_iterator(x, y, batch_size, shuffle=False):
    if y is not None:
        assert x.shape[0]==y.shape[0], 'x and y not equal size'
    if shuffle:
        indices=np.random.permutation(x.shape[0])
        x=x[indices,:]
        y=y[indices,:] if y is not None else None
    
    data_len=x.shape[0]
    batch_len=data_len//batch_size
    for i in xrange(batch_len):
        _x=x[i*batch_size:(i+1)*batch_size,:]
        _y=None
        if y is not None:
            _y=y[i*batch_size:(i+1)*batch_size,:]
        yield(_x,_y)
    i=batch_len
    _x=x[i*batch_size:,:]
    _y=None
    if y is not None:
        _y=y[i*batch_size:,:]
    yield(_x,_y)

class Config():

    def __init__(self):
        self.batch_size=64
        self.hidden_size=[30,20,10,1]
        self.max_epochs=100
        self.early_stopping=100
        self.dropout=1.
        self.lr=0.0001
        self.input_size=170

    def to_dict(self):
        dic={
            'batch_size':self.batch_size,
            'hidden_size':self.hidden_size,
            'max_epochs':self.max_epochs,
            'early_stopping':self.early_stopping,
            'dropout':self.dropout,
            'lr':self.lr,
            'input_size':self.input_size
        }
        return dic

class Model():
    
    def load_data(self, train_x, train_y, val_x, val_y, test_x, test_y=None):
        self.train_x=train_x
        self.train_y=train_y
        self.val_x=val_x
        self.val_y=val_y
        self.test_x=test_x
        self.test_y=test_y
 
    def add_placeholders(self):
        self.input_placeholder=tf.placeholder(tf.float32,shape=[None,self.config.input_size])
        self.labels_placeholder=tf.placeholder(tf.float32,shape=[None,1])
        self.dropout_placeholder=tf.placeholder(tf.float32)

    def add_model(self, inputs):
        xavier_initializer=xavier_weight_init()
        h_size=self.config.hidden_size
        inp=inputs
        in_sz=self.config.input_size
        for sz in h_size[:-1]:
            W=tf.Variable(xavier_initializer([in_sz,sz]))
            b=tf.Variable(xavier_initializer([sz]))
            hidden=tf.sigmoid(tf.matmul(tf.nn.dropout(inp,self.dropout_placeholder),W)+b)
            inp=hidden
            in_sz=sz
        
        W=tf.Variable(xavier_initializer([in_sz,h_size[-1]]))
        b=tf.Variable(xavier_initializer([h_size[-1]]))
        output=tf.matmul(tf.nn.dropout(inp,self.dropout_placeholder),W)+b
        return output

    def add_loss_op(self, y):
        sub=y-self.labels_placeholder
        weight=tf.mul(self.labels_placeholder,self.labels_placeholder)
        loss=tf.reduce_mean(tf.div(tf.mul(sub,sub),weight))
        return loss

    def add_training_op(self, loss):
        train_op=tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def __init__(self, config, train_x, train_y, val_x, val_y, test_x, test_y=None):
        self.config=config
        self.load_data(train_x, train_y, val_x, val_y, test_x, test_y)
        self.add_placeholders()
        self.predictions=self.add_model(self.input_placeholder)
        self.loss=self.add_loss_op(self.predictions)
        self.train_step=self.add_training_op(self.loss)

    def run_epoch(self, session, x, y=None, train_op=None, shuffle=True, verbose=10):
        dp=self.config.dropout
        predictions=self.predictions
        loss=self.loss
        if not train_op:
            train_op=tf.no_op()
            dp=1
        if y is None:
            loss=tf.no_op()

        total_steps=sum(1 for x in data_iterator(x, y, self.config.batch_size))
        total_loss=[]
        total_pred=[]

        for step, (_x, _y) in enumerate(data_iterator(x, y, self.config.batch_size, shuffle)):
            feed={self.input_placeholder: _x,
                  self.dropout_placeholder: dp}
            if _y is not None:
                feed[self.labels_placeholder]=_y
            
            _pred, _loss, _=session.run([predictions, loss, train_op], feed_dict=feed)
            total_pred.append(_pred)
            if y is not None:
                total_loss.append(_loss)
            if verbose and step % verbose==0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()

        assert np.vstack(total_pred).reshape([-1]).shape[0]==x.shape[0], 'pred and x not equal size'
        return np.vstack(total_pred).reshape([-1]), np.mean(total_loss)

def nn_solver(train_x, train_y, validation_x, test_x, now_time , validation_y = np.array([]), feature_names = [], debug=False):
    """
    """
    if debug:
        validation_x=train_x
        validation_y=train_y
    
    logging.info('start training the neural network model')
    
    config=Config()
    config.input_size=train_x.shape[1]

    with open(ROOT + '/result/' + now_time + '/parameters.param', 'w') as out :
        for key, val in config.to_dict().items():
            out.write(str(key) + ': ' + str(val) + '\n')

    model=Model(config, train_x, train_y.reshape([-1,1]), validation_x, validation_y.reshape([-1,1]), test_x)
    init=tf.initialize_all_variables()
    saver=tf.train.Saver()

    session=tf.Session()
    session.run(init)

    best_val_validate = float('inf')
    best_val_epoch = 0
    best_val_y = None

    train_loss_total=[]
    validation_loss_total=[]

    for epoch in xrange(config.max_epochs):
        logging.info('Epoch{}'.format(epoch))
        
        train_loss_total.append(
            model.run_epoch(session, model.train_x, model.train_y, train_op=model.train_step)[1])
        val_y, val_loss =model.run_epoch(session, model.val_x, model.val_y, shuffle=False, verbose=None)
        validation_loss_total.append(val_loss)
        logging.info('Train Loss: {}'.format(train_loss_total[-1]))
        logging.info('Validation Loss: {}'.format(validation_loss_total[-1]))
        if validation_loss_total[-1] < best_val_validate:
            best_val_validate=validation_loss_total[-1]
            best_val_epoch=epoch
            best_val_y=val_y
            saver.save(session, ROOT+'/result/'+now_time+'/model/nn.weights')
        if epoch - best_val_epoch > config.early_stopping:
            break
    
#    saver.restore(session, ROOT+'/result/'+now_time+'/model/nn.weights')
    test_y=model.run_epoch(session, model.test_x, shuffle=False, verbose=None)[0]

    if validation_y.shape[0]  :
        logging.info('the loss in Training set is %.4f' % train_loss_total[best_val_epoch])
        logging.info('the loss in Validation_set is %.4f' % validation_loss_total[best_val_epoch])

        plt.figure(figsize=(12, 6))
        # No feature importance

        # Plot training deviance
        #plt.subplot(1, 2, 2)
        plt.title('Deviance')
        plt.plot(np.arange(epoch+1) + 1, train_loss_total, 'b-',
                          label='Training Set Deviance')
        plt.plot(np.arange(epoch+1) + 1, validation_loss_total, 'r-',
                          label='Validation Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')

        plt.savefig(ROOT + '/result/' + now_time + '/statistics.jpg')

        print "------best_val_y------"
        print best_val_y.reshape([-1,1])
        print np.max(best_val_y),np.min(best_val_y)
        print '------val_y-----------'
        print validation_y.reshape([-1,1])
        print '------val_x-----------'
        print validation_x
        print '------sum_abs_--------'
        print np.sum(np.abs(best_val_y-validation_y))

        print "not zero prediction : %d " % sum( [ i!=0 for i in best_val_y.astype(int).tolist()] )
        print "total number of train data : %d" % train_y.shape[0]
        print "not zero label train data : %d" % sum(train_y!=0)
        print "total number of validation data : %d" % validation_y.shape[0]
        print "not zero label validation data : %d" % sum(validation_y!=0)

    return best_val_y, test_y

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

    solver.main(nn_solver, type = type, dimreduce_func = feature_reduction.undo) 
    solver.main(nn_solver, gap_month=2, type=type, dimreduce_func = feature_reduction.undo)
    evaluate.mergeoutput()
