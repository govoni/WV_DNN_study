#!/usr/local/bin/python3

import argparse 
import configparser
from utils import *

if __name__ == "__main__":

  parser = argparse.ArgumentParser ()
  parser.add_argument ("-c","--config", help="configuration file", type=str)
  args = parser.parse_args ()

  config = configparser.ConfigParser ()
  config.read (args.config)

  createOutputFolder (config)

