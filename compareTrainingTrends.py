#!/usr/local/bin/python3

import numpy as np
import argparse 
import configparser
from matplotlib import pyplot as plt
from utils import *
from plotTrainingTrends import *


# ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ====


if __name__ == "__main__" :

  results_DO0p00_100e, epochs_DO0p00_100e, foms_DO0p00_100e = getHistory ('prova_DO0p00_100e-001')
  results_DO0p10_100e, epochs_DO0p10_100e, foms_DO0p10_100e = getHistory ('prova_DO0p10_100e-001')

  results_DO0p00_50e, epochs_DO0p00_50e, foms_DO0p00_50e = getHistory ('prova_DO0p00_50e-001')
  results_DO0p10_50e, epochs_DO0p10_50e, foms_DO0p10_50e = getHistory ('prova_DO0p10_50e-001')

  plots = [fom for fom in foms_DO0p00_100e if not (fom.startswith ('val'))]

  ylimits = {
    'loss'                        : (3.70e-5, 4.20e-5), 
    'binary_crossentropy'         : (0.60, 0.65), 
    'accuracy'                    : (0.64, 0.68), 
    'kullback_leibler_divergence' : (0.19, 0.26), 
    'auc'                         : (0.72, 0.76),
    }

  x = range (epochs_DO0p00_100e)
  x_50 = range (epochs_DO0p10_50e)
  # loop over plots to be created
  for plot in plots :
    addTrend (plt, x, results_DO0p00_100e, plot, 'dodgerblue', 'train no DO 100')
    addTrend (plt, x, results_DO0p00_100e, 'val_' + plot, 'dodgerblue', 'test no DO 100', 2)
    # addTrend (plt, x, results_DO0p10_100e, plot, 'coral', 'train DO=0.1 100')
    # addTrend (plt, x, results_DO0p10_100e, 'val_' + plot, 'coral', 'test DO=0.1 100', 2)
    addTrend (plt, x_50, results_DO0p00_50e, plot, 'limegreen', 'train no DO 50')
    addTrend (plt, x_50, results_DO0p00_50e, 'val_' + plot, 'limegreen', 'test no DO 50', 2)
    plt.ylim (ylimits[plot])
    plt.legend ()
    plt.ylabel (plot)
    plt.xlabel ('epoch')
    plt.grid (which = 'major', axis = 'y')
    plt.savefig ('FOM_' + plot + '.png')
    plt.clf ()
    # END - loop over plots to be created

