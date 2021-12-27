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

# plot background distributions
# -------------------

# loop over variables to be plotted
for var in config['study']['variables'].split ():
  if (var == 'weight') : continue
  print ('python che pazienza ' + var)
  hmin = float (config['plot'][var].split ()[1])
  hmax = float (config['plot'][var].split ()[2])
  step = float ((hmax - hmin) / float (config['plot'][var].split ()[0]))
  pazienza = plt.hist (tuple ([df[1][var] for df in bkgs]), 
               # density = True ,  # normalised distribution
               histtype  = 'stepfilled', 
               color     = [config['plot'][sam].split ()[0] for sam in config['input']['bkg'].split ()] ,
# questo non funziona
#               edgecolor = [config['plot'][sam].split ()[1] for sam in config['input']['bkg'].split ()] ,
               weights   = tuple ([sam[1]['weightTN'] for sam in bkgs]) ,
               range     = (hmin, hmax) ,
               bins      = np.arange (hmin, hmax, step) ,
               label     = [sam[0] for sam in bkgs] ,
               )
  if (len (config['plot'][var].split ()) == 4): plt.xlabel (var + ' (' + config['plot'][var].split ()[3] + ')')
  else                                        : plt.xlabel (var)
  plt.legend (loc = 'upper right')
  plt.yscale (config['plot']['yscale'])
  plt.savefig (var + '_bkg.png')
  plt.clf ()
  # END - loop over variables to be plotted


# https://stackoverflow.com/questions/51298952/bin-counts-in-stacked-histogram-weighted-with-x-coordinate-greater-than-certai

