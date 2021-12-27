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

# get the backgrounds
# -------------------

bkgs = []
index = 0
# loop over bkg samples
for sample in config['input']['bkg'].split () :
  df_B = pd.read_csv ("{}/dataframe-{}-{}.csv".format(config['output']['dfdir'],cat,sample),header=0)
  # name, dataframe, XS
  bkgs.append ([sample, df_B, config['input']['bkg_XS'].split ()[index]])
  index = index + 1
  # END loop over bkg samples

# plot distributions
# -------------------

# loop over variables to be plotted
for var in config['study']['variables'].split ():
  if (var == 'weight') : continue
  print ('python che pazienza ' + var)
  hmin = float (config['plot'][var].split ()[1])
  hmax = float (config['plot'][var].split ()[2])
  step = float ((hmax - hmin) / float (config['plot'][var].split ()[0]))

  # loop over bgk samples
  for sam in bkgs:
    paz = plt.hist (sam[1][var], 
           density   = True ,  # normalised distribution
           histtype  = 'stepfilled' , 
           color     = config['plot'][sam[0]].split ()[0] ,
           edgecolor = config['plot'][sam[0]].split ()[1],
           weights   = sam[1]['weightTN'] ,
           range     = (hmin, hmax) ,
           bins      = np.arange (hmin, hmax, step) ,
           label     = sam[0] ,
           fill      = False ,
           )
    # END - loop over bkg samples

  paz = plt.hist (df_S[var], 
         density   = True ,  # normalised distribution
         histtype  = 'stepfilled', 
         color     = config['plot']['VBS'].split ()[0],
         edgecolor = config['plot']['VBS'].split ()[1],
         weights   = [w for w in df_S['weightTN']] ,
         range     = (hmin, hmax) ,
         bins      = np.arange (hmin, hmax, step) ,
         label     = 'VBS' ,
         fill      = False ,
         )

  if (len (config['plot'][var].split ()) == 4): plt.xlabel (var + ' (' + config['plot'][var].split ()[3] + ')')
  else                                        : plt.xlabel (var)
  plt.legend (loc = 'upper right')
  plt.yscale (config['plot']['yscale'])
  plt.savefig (var + '_cfrnorm.png')
  plt.clf ()
  # END - loop over variables to be plotted


# https://stackoverflow.com/questions/51298952/bin-counts-in-stacked-histogram-weighted-with-x-coordinate-greater-than-certai

