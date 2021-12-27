#!/usr/local/bin/python3

# from https://github.com/UniMiBAnalyses/PlotsConfigurations/blob/VBSjjlnu_v7/Configurations/VBSjjlnu/scripts/extract_pandas_df.py

import pandas as pd 
import numpy as np
import ROOT as R 
import sys
import configparser
import argparse 
from pprint import pprint
import os


def calc_weight (row, global_factor):
  return row['weight'] * global_factor


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


if __name__ == "__main__":

  parser = argparse.ArgumentParser ()
  parser.add_argument ("-c","--config", help="configuration file", type=str)
  args = parser.parse_args ()

  config = configparser.ConfigParser ()
  config.read (args.config)

  # non so se questa istruzione cancelli una cartella se già esiste, 
  # quindi per ora non è attiva
  # os.makedirs (args.outputdir, exist_ok=True)

  samples = config['input']['signal'].split () + config['input']['bkg'].split () 
  xss     = config['input']['signal_XS'].split () + config['input']['bkg_XS'].split () 
  eff     = config['input']['signal_eff'].split () + config['input']['bkg_eff'].split () 

  R.ROOT.EnableImplicitMT();

  cat = config['input']['category']

  # loop over the various samples
  index = int (0)
  for sample in samples:
    print('  ' + sample)
    rdf = R.RDataFrame (cat + "/" + config['input']['label'] + "/tree_" + sample, config['input']['rootfile'])
    df = pd.DataFrame (rdf.AsNumpy())

    # add a weight to be used to get the number of expected events (1000. accounts for pb --> fb conversion)
    global_factor = float (config['input']['lumi']) * 1000. * float (xss[index]) * float (eff[index]) / df['weight'].sum ()
    df['weightTN'] = df.apply (lambda row : calc_weight (row, global_factor), axis=1)

    df.to_csv ("{}/dataframe-{}-{}.csv".format(config['output']['dfdir'],cat,sample), index=False, sep=",")
    index = index + 1

    # END - loop over the various samples

  # END - main function