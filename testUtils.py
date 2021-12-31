#!/usr/local/bin/python3

import argparse 
import configparser
from utils import *

def testOutputFolder () :
  parser = argparse.ArgumentParser ()
  parser.add_argument ("-c","--config", help="configuration file", type=str)
  args = parser.parse_args ()

  config = configparser.ConfigParser ()
  config.read (args.config)

  createOutputFolder (config)


# ---- ---- ---- ---- ---- ---- ---- ---- ----



def testSavedHistory (filename) :
  with open(filename, "rb") as f:
    history = pickle.load (f)
    print (history.keys ())



# ==== ==== ==== ==== ==== ==== ==== ==== ====


if __name__ == "__main__":

#  testOutputFolder ()
  print (getNextName ('DNNhistory', 'prova-001'))
  testSavedHistory ('./prova-001/DNNhistory-001')



