#!/usr/local/bin/python3

import argparse 

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

#from tensorflow import keras 
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers.schedules import InverseTimeDecay



def run (X, Y, W) :

  # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
  X_train, X_test, Y_train, Y_test, W_train, W_test = \
    train_test_split (
      X, Y,  W, 
      test_size    = 0.5,    # fraction of samples used for testing
      random_state = 41,     # equivalent to the random seed
      stratify     = Y       # useful with several classes of samples, it seems
    )

  # build the DNN model
  # -------------------

  # the DNN model
  #  - https://www.tensorflow.org/api_docs/python/tf/keras/Model
  model = Sequential ()
  # add input layer
  model.add (Dense (5, input_dim = 2 , activation = 'relu'))
  model.add (Dense (5, activation = 'relu'))
  model.add (Dense (1, activation = 'sigmoid'))

  model.compile (
      optimizer = tf.keras.optimizers.Adam () ,
      loss      = 'binary_crossentropy',
    )

  history = model.fit (
        X_train, Y_train,
        sample_weight   = W_train ,
        epochs          = 10 ,
        validation_data = (X_test, Y_test, W_test) ,
        callbacks       = [] ,
        shuffle         = True ,
        batch_size      = 32 ,
        verbose         = False ,
    )

  return history


# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====


if __name__ == "__main__" :

  parser = argparse.ArgumentParser ()
  parser.add_argument ("-w0","--weight0", help="weight for sample H0", type = float)
  parser.add_argument ("-w1","--weight1", help="weight for sample H1", type = float)
  parser.add_argument ("-N","--Nevt", help="number of events", type = int)
  args = parser.parse_args ()

  # input samples generation and visualisation
  # ------------------------------------------

  ave_H0 = np.array ([0., 0.])
  cov_H0 = np.array ([[1., 0.3], [0.3 ,1. ]])
  s_H0 = np.random.multivariate_normal (ave_H0, cov_H0, args.Nevt)
  w_H0 = np.full (args.Nevt, args.weight0)
  y_H0 = np.full (args.Nevt, 0.)

  ave_H1 = np.array ([3., 2.])
  cov_H1 = np.array ([[1., -0.5], [-0.5 ,1. ]])
  s_H1 = np.random.multivariate_normal (ave_H1, cov_H1, args.Nevt)
  w_H1 = np.full (args.Nevt, args.weight1)
  y_H1 = np.full (args.Nevt, 1)

  fig, axs = plt.subplots (2, 2)
  fig.tight_layout(pad = 2.0)

  sns.kdeplot (
      x     = s_H0[:,0] , 
      y     = s_H0[:,1] ,
      color = 'coral' ,
      ax    = axs[0,0] ,
      label = 'H0' ,
      fill  = True ,
    )

  sns.kdeplot (
      x     = s_H1[:,0] , 
      y     = s_H1[:,1] ,
      color = 'dodgerblue' ,
      ax    = axs[0,0] ,
      label = 'H1' ,
    )

  axs[0,0].set (xlabel = 'x')
  axs[0,0].set (ylabel = 'y')
  axs[0,0].legend ()

  axs[1,0].hist (s_H0[:,0], 
     histtype  = 'stepfilled', 
     color     = 'coral' ,
     weights   = w_H0 ,
     range     = (-4, 6) ,
     bins      = np.arange (-4, 6, 1) ,
     label     = 'H0' ,
   )

  axs[1,0].hist (s_H1[:,0], 
     histtype  = 'stepfilled', 
     edgecolor = 'dodgerblue' ,
     linewidth = 2 ,
     fill      = False ,
     weights   = w_H1 ,
     range     = (-4, 6) ,
     bins      = np.arange (-4, 6, 1) ,
     label     = 'H1' ,
   )

  axs[1,0].set (xlabel = 'x')
  axs[1,0].legend ()

  axs[0,1].hist (s_H0[:,1], 
     histtype  = 'stepfilled', 
     color     = 'coral' ,
     weights   = w_H0 ,
     range     = (-3, 7) ,
     bins      = np.arange (-3, 7, 1) ,
     label     = 'H0' ,
   )

  axs[0,1].hist (s_H1[:,1], 
     histtype  = 'stepfilled', 
     edgecolor = 'dodgerblue' ,
     linewidth = 2 ,
     fill      = False ,
     weights   = w_H1 ,
     range     = (-3, 7) ,
     bins      = np.arange (-3, 7, 1) ,
     label     = 'H1' ,
   )

  axs[0,1].set (xlabel = 'y')
  axs[0,1].legend ()

  axs[1,1].hist (w_H0, 
     histtype  = 'stepfilled', 
     color     = 'coral' ,
     label     = 'H0' ,
   )

  axs[1,1].hist (w_H1, 
     histtype  = 'stepfilled', 
     edgecolor = 'dodgerblue' ,
     linewidth = 2 ,
     fill      = False ,
     label     = 'H1' ,
   )

  axs[1,1].set (xlabel = 'weight')
  axs[1,1].legend ()
 
  fig.savefig ('CW_' + str (args.weight0) + '_' + str (args.weight1) + '_' + str (args.Nevt) + '_init.png')
  plt.clf ()

  # DNN model preparation and run
  # ------------------------------------------

  X     = np.vstack ([s_H0, s_H1])
  Y     = np.hstack ([y_H0, y_H1])
  W     = np.hstack ([w_H0, w_H1])

  nToys = 100

  loss = []
  for iToy in range (nToys) :
    print ('toy', iToy)
    history = run (X, Y, W)
    loss.append (history.history['loss'][-1])

  plt.hist (loss)
  fig.savefig ('CW_' + str (args.weight0) + '_' + str (args.weight1) + '_' + str (args.Nevt) + '_loss.png')

