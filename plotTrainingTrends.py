#!/usr/local/bin/python3

import numpy as np
import argparse 
import configparser
from matplotlib import pyplot as plt
from utils import *


def getHistory (folderName) :
  print ('reading', folderName)

  historyFileNames = [folderName + '/' + elem for elem in os.listdir (folderName) if elem.startswith ('DNNhistory')]
  
  toys = len (historyFileNames)
  print ('analysing', toys, 'history files')

  epochs = int (0)
  foms = [] # figures of merit
  with open (historyFileNames[0], 'rb') as f0:
    history = pickle.load (f0)
    for key in history.keys () :
      foms.append (key)
    epochs = len (next (iter (history.values ())))  
  print (foms)
  print (epochs)

  results = {}
  # loop over history files
  for fom in foms : results[fom] = [np.zeros (epochs), np.zeros (epochs)]
  for file in historyFileNames :
    with open (file, 'rb') as f :
      history = pickle.load (f)
      for key in history.keys () :
        npa = np.asarray (history[key], dtype=np.float32)
        results[key][0] = results[key][0] + npa
        results[key][1] = results[key][1] + (npa * npa)
    # END - loop over history files

  for key in results:
    results[key][0] = results[key][0] / toys
    results[key][1] = np.sqrt (results[key][1] / toys - results[key][0] * results[key][0])
  return results, epochs, foms


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def addTrend (plt, xvals, results, plot, color, label) :
  # simple error bars
  # plt.errorbar (x, results[plot][0], 
  #               yerr = results[plot][1], label = label, color = color)
  # filled area with white line in the middle
  plt.fill_between (x,  results[plot][0] - results[plot][1], 
                   results[plot][0] + results[plot][1], 
                   color = color, label = label, alpha = 0.5)
#  plt.plot (x, results[plot][0], color = 'white')
  plt.plot (x, results[plot][0], color = color)
  # three lines: extreme ones and central one
  # plt.plot (x, results[plot][0], color = color, label = label)
  # plt.plot (x, results[plot][0] - results[plot][1], color = color)
  # plt.plot (x, results[plot][0] + results[plot][1], color = color)


# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====


if __name__ == "__main__":

  parser = argparse.ArgumentParser ()
  parser.add_argument ("-f","--folder", help="results folder", type=str)
  args = parser.parse_args ()

  results, epochs, foms = getHistory (args.folder)

  plots = [fom for fom in foms if not (fom.startswith ('val'))]

  ylimits = {
    'loss'                        : (3.70e-5, 4.20e-5), 
    'binary_crossentropy'         : (0.60, 0.65), 
    'accuracy'                    : (0.64, 0.68), 
    'kullback_leibler_divergence' : (0.19, 0.26), 
    'auc'                         : (0.72, 0.76),
    }

  x = range (epochs)
  # loop over plots to be created
  for plot in plots :
    addTrend (plt, x, results, plot, 'lightskyblue', 'train')
    addTrend (plt, x, results, 'val_' + plot, 'coral', 'test')
    plt.ylim (ylimits[plot])
    plt.legend ()
    plt.ylabel (plot)
    plt.xlabel ('epoch')
    plt.savefig (args.folder + '/FOM_' + plot + '.png')
    plt.clf ()
    # END - loop over plots to be created

