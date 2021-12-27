#!/usr/local/bin/python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import configparser
import argparse 
import sys
import os

parser = argparse.ArgumentParser ()
parser.add_argument ("-c","--config", help="configuration file", type=str)
args = parser.parse_args ()

config = configparser.ConfigParser ()
config.read (args.config)

cat = 'boost_sig_mu'

# get the signal
# -------------------

signals = []
df_S = pd.read_csv ("{}/dataframe-{}-{}.csv".format(config['output']['dfdir'],cat,'VBS'),header=0)
signals.append (['VBS', df_S, config['input']['signal_XS'].split ()[0]])

# plot signal distributions
# -------------------

for var in config['study']['variables'].split ():
  if (var == 'weight') : continue
  print ('python che pazienza ' + var)
  hmin = float (config['plot'][var].split ()[1])
  hmax = float (config['plot'][var].split ()[2])
  step = float ((hmax - hmin) / float (config['plot'][var].split ()[0]))
  pazienza = plt.hist (df_S[var], 
               # density = True ,  # normalised distribution
               histtype  = 'stepfilled', 
               color     = config['plot']['VBS'].split ()[0],
               edgecolor = config['plot']['VBS'].split ()[1],
               weights   = df_S['weightTN'] ,
               # legend   = False , # the pandas df legend looks like done by excel...
               range     = (hmin, hmax) ,
               bins      = np.arange (hmin, hmax, step) ,
               )
  plt.savefig (var + '_sig.png')
  plt.clf ()

