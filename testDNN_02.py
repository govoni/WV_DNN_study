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


def readSample (pd, config, cat, sample) :
  df = pd.read_csv ("{}/dataframe-{}-{}.csv".format (config['output']['dfdir'], cat, sample), header=0)
  X = df [config['study']['trainvars'].split ()]
  Wtot = df ['weightTN'].sum ()
  W = [w / Wtot for w in df ['weightTN']]
  Wnn = df ['weightTN']
  print ('read ', sample, ': ', df['weightTN'].sum ())
  return X, W, Wnn


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def compareSamples (lista, N_evt, N_var, label) :
  # loop over vars
  for iVar in range (3):
    sig = [e[iVar] for e in lista[:N]]
    bkg = [e[iVar] for e in lista[N:]]
    plt.hist (sig, bins = 20, histtype  = 'stepfilled', fill = False, edgecolor = 'red')
    plt.hist (bkg, bins = 20, histtype  = 'stepfilled', fill = False, edgecolor = 'blue')
    plt.savefig ('var_' + str (iVar) + label + '.png')
    plt.clf ()
    # END - loop over vars
  return


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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

  from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

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

  from sklearn.model_selection import train_test_split
  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
  X_train, X_test, y_train, y_test, W_train, W_test , Wnn_train, Wnn_test = \
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

  import tensorflow as tf

  #from tensorflow import keras 
  from tensorflow.keras.models import Sequential
  from tensorflow.keras import regularizers
  from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout , LeakyReLU

  DNNlayers = config['study']['DNNlayers'].split ()
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

  print (model.summary ())