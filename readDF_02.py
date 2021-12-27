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

df_S = pd.read_csv ("{}/dataframe-{}-{}.csv".format(config['output']['dfdir'],cat,'VBS'),header=0)

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
               range     = (hmin, hmax) ,
               bins      = np.arange (hmin, hmax, step) ,
               label     = 'VBS' ,
               )
  if (len (config['plot'][var].split ()) == 4): plt.xlabel (var + ' (' + config['plot'][var].split ()[3] + ')')
  else                                        : plt.xlabel (var)
  plt.legend (loc = 'upper right')
  plt.yscale (config['plot']['yscale'])
  plt.savefig (var + '_sig.png')
  plt.clf ()

