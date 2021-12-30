#!/usr/local/bin/python3

# https://github.com/UniMiBAnalyses/ML_classification/blob/master/Training_v3_resolved_weights_Aurora.ipynb

import argparse 
import configparser
import pandas as pd 
import sys
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
    
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

#from tensorflow import keras 
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers.schedules import InverseTimeDecay

from utils import *


if __name__ == "__main__":

  parser = argparse.ArgumentParser ()
  parser.add_argument ("-c","--config", help="configuration file", type=str)
  args = parser.parse_args ()

  config = configparser.ConfigParser ()
  config.read (args.config)

  cat = 'boost_sig_mu'

  # get the signal and background samples
  # -------------------

  X_sig, W_sig, Wnn_sig = readSample (pd, config, cat, 'VBS')
  sample = config['input']['bkg'].split ()[0]
  X_bkg, W_bkg, Wnn_bkg = readSample (pd, config, cat, sample)

  # get the number of envents in the signal and in the background samples
  # -------------------

  N = min (X_sig.shape[0], X_bkg.shape[0])
  print ('events number per sample: ', N)
 
  # transform the input variables
  # -------------------

  #  - https://scikit-learn.org/stable/modules/preprocessing.html

  # concatenate signal and bkg, so that both are transformed in the same way
  X = np.vstack([X_sig[:N], X_bkg[:N]])
  scaler = StandardScaler ()
  X_scaled = scaler.fit_transform (X)

  # additional containers needed for training
  #  - NB the training sample is a single one, containing signal (y==1) and bkg (y==0)
  # -------------------

  Y_sig = np.ones (len (X_sig))
  Y_bkg = np.zeros (len (X_bkg))
  Y     = np.hstack ([Y_sig[:N], Y_bkg[:N]])
  W     = np.hstack ([W_sig[:N], W_bkg[:N]])
  Wnn   = np.hstack ([Wnn_sig[:N], Wnn_bkg[:N]])

  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
  X_train, X_test, Y_train, Y_test, W_train, W_test, Wnn_train, Wnn_test = \
    train_test_split (
      X_scaled, Y,  W, Wnn, 
      test_size    = 0.5,    # fraction of samples used for testing
      random_state = 41,     # equivalent to the random seed
      stratify     = Y       # useful with several classes of samples, it seems
    )

  print ('Training   dataset: ', X_train.shape)
  print ('Test + Val dataset: ', X_test.shape)

  # build the DNN model
  # -------------------

  DNNlayers = config['study']['DNNlayers'].split ()

  # the DNN model
  #  - https://www.tensorflow.org/api_docs/python/tf/keras/Model
  model = Sequential ()
  model.add (Dense (
      config['study'][DNNlayers[0]].split ()[0], 
      input_dim = len (config['study']['trainvars'].split ()) ,
      activation = config['study'][DNNlayers[0]].split ()[1] ,
    ))
  # loop over additional layers
  for iLayer in range (1, len (DNNlayers)) :
    model.add (Dense (
        config['study'][DNNlayers[iLayer]].split ()[0], 
        activation = config['study'][DNNlayers[iLayer]].split ()[1] ,
      ))
    # END - loop over additional layers

  model.compile (
      optimizer = config['study']['DNNoptimizer'] ,
      loss      = config['study']['DNNloss'] ,
      metrics   = config['study']['DNNmetrics'].split () ,
    )

  print (model.summary ())

  # train the DNN model
  # -------------------

  # callbacks 
  #  - A callback is an object that can perform actions 
  #    at various stages of training (e.g. at the start or end of an epoch, 
  #    before or after a single batch, etc).
  #  - https://keras.io/api/callbacks/
  #  - https://blog.paperspace.com/tensorflow-callbacks/

  early_stopping = callbacks.EarlyStopping (
      monitor              = 'val_loss', 
      min_delta            = float (config['study']['DNNEarlyStop'][1]), 
      patience             = float (config['study']['DNNEarlyStop'][0]), 
      verbose              = 0, 
      mode                 = 'auto', 
      baseline             = None, 
      restore_best_weights = True
    )

  # LearningRateScheduler or ReduceLROnPlateau may become a useful tool

  used_callbacks = []
#  used_callbacks = [early_stopping]

  history = model.fit (
        X_train, Y_train,
        sample_weight   = W_train,
        epochs          = int (config['study']['DNNepochs']),
        validation_data = (X_test, Y_test, W_test),
        callbacks       = used_callbacks,
        shuffle         = True,
        batch_size      = int (config['study']['DNNbatchSize']),
        #  class_weight= {0:1.8,1:1}
        verbose         = False ,
    )

  plotMetric (history, 'loss')
  for metric in config['study']['DNNmetrics'].split ():
    plotMetric (history, metric.lower ())




