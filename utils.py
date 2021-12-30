# general utility functions


def getOutputFolderName (tag):
  folders = [elem for elem in os.listdir ('./') if os.path.isdir (elem) if elem.startswith (tag)]
  num = int (1)
  if (len (folders) > 0) :
    num = sorted ([int (elem.split ('-')[-1]) for elem in folders])[-1] + 1
  return tag + '-' + str (num).zfill (3)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def readSample (pd, config, cat, sample) :
  df = pd.read_csv ("{}/dataframe-{}-{}.csv".format (config['output']['dfdir'], cat, sample), header=0)
  X = df [config['study']['trainvars'].split ()]
  Wtot = df ['weightTN'].sum ()
  W = [w / Wtot for w in df ['weightTN']]
  Wnn = df ['weightTN']
  print ('read ', sample, ': ', df['weightTN'].sum ())
  return X, W, Wnn


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def compareSamples (lista, N_evt, N_var, label) :
  # loop over vars
  for iVar in range (3):
    sig = [e[iVar] for e in lista[:N]]
    bkg = [e[iVar] for e in lista[N:]]
    plt.hist (sig, bins = 20, histtype  = 'stepfilled', fill = False, edgecolor = 'red')
    plt.hist (bkg, bins = 20, histtype  = 'stepfilled', fill = False, edgecolor = 'blue')
    plt.savefig ('var_' + str (iVar) + label + '.png')
    plt.clf ()
    # END - loop over vars
  return


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def plotMetric (history, metric, outFolder = './'):
  fig, ax1 = plt.subplots (figsize=(7,6), dpi=100)
  plt.plot (history.history[metric], label='train')
  plt.plot (history.history['val_' + metric], label='val')
  plt.ylabel (metric)
  plt.xlabel ('epoch')
  plt.legend (loc = 'upper right')
  plt.savefig (outfolder + '/DNNtraining_' + metric + '.png')
  plt.clf ()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def ave (lista):
  tot = 0.
  for i in lista : tot = tot + float (i)
  return tot / len (lista)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def getOTdifference (history, metric):
  return ave (history.history['val_' + metric][-5:]) \
         - ave (history.history[metric][-5:])


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

