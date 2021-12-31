#!/usr/local/bin/python3

import argparse 
import configparser
from utils import *


if __name__ == "__main__":

  parser = argparse.ArgumentParser ()
  parser.add_argument ("-f","--file", help="history file", type=str)
  args = parser.parse_args ()

  print ('reading', args.file)

  with open(args.file, "rb") as f:
    history = pickle.load (f)
    for key in history.keys ():
      if key.startswith ('val') : continue
      plotMetric (history, key, outFolder = './')




