#!/usr/local/bin/python3

import pandas as pd 
import numpy as np
import ROOT as R 
import sys
import configparser
import argparse 
from pprint import pprint
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
print ('read VBS: ', df_S['weightTN'].sum ())

# get the backgrounds
# -------------------

bkgs = []
index = 0
# loop over bkg samples
for sample in config['input']['bkg'].split ():
  df_B = pd.read_csv ("{}/dataframe-{}-{}.csv".format(config['output']['dfdir'],cat,sample),header=0)
  bkgs.append ([sample, df_B, config['input']['bkg_XS'].split ()[index]])
  print ('read ' + sample + ': ' + df_B['weightTN'].sum ())
  index = index + 1
  # END loop over bkg samples



