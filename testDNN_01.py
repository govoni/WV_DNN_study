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

  # get the signal
  # -------------------

  # name, features, weights
  sigs = []
  df_S = pd.read_csv ("{}/dataframe-{}-{}.csv".format(config['output']['dfdir'],cat,'VBS'),header=0)
  sigs.append (['VBS', df_S, config['input']['signal_XS'].split ()[0]])
  print ('read VBS: ', df_S['weightTN'].sum ())
  X_sig = df_S [config['study']['trainvars'].split ()]
  sigs.append (['VBS', X_sig, df_S['weightTN']])

  # get the backgrounds
  # -------------------

  bkgs = []
  # loop over bkg samples
  sample = config['input']['bkg'].split ()[0]
  df_B = pd.read_csv ("{}/dataframe-{}-{}.csv".format(config['output']['dfdir'],cat,sample),header=0)
  X_bkg = df_B [config['study']['trainvars'].split ()]
  bkgs.append ([sample, X_bkg, df_B['weightTN']])
  print ('read ', sample, ': ', df_B['weightTN'].sum ())

  # get the number of envents in the signal and in the background samples
  # -------------------

  N = min (sigs[0][1].shape[0], bkgs[0][1].shape[0])
  print ('minimum common number of events', N)

  # transform the input variables
  # -------------------

  #  - https://scikit-learn.org/stable/modules/preprocessing.html

  from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

  # concatenate signal and bkg, so that both are transformed in the same way
  X = np.vstack([X_sig[:N], X_bkg[:N]])
  # visualize before scaling the signal and the background distributions
  compareSamples (X, N, 3, 'pre')
  print ('mean and standard deviation before scaling')
  print ('M: ',X.mean (axis=0))
  print ('S: ',X.std (axis=0))

  scaler = StandardScaler ()
  X_scaled = scaler.fit_transform (X)
  # visualize after scaling the signal and the background distributions
  compareSamples (X_scaled, N, 3, 'post')
  print ('mean and standard deviation after scaling')
  print ('M: ',X_scaled.mean (axis=0))
  print ('S: ',X_scaled.std (axis=0))


