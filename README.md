# DNN for QML studies

## content of the folder

  * ```params*.cfg```: input config files for the data analysis
  * ```utils.py```: generic utility functions
  * ```testUtils.py```: testing the utils library
  * ```readNtuple.py```: translate root ntuples into Pandas
    * usage: ```./readNtuple.py -c readNtuple.cfg```
  * ```readDF_01.py```: for one single category, 
    re-read the dataframes saved on file by ```readNtuple.py```
    and put the dataframes in lists (or lists of lists),
    associating each dataframe to its name and to the corresponding cross-section
  * ```readDF_02.py```: produce simple signal histograms from the pandas dataframe
  * ```readDF_03.py```: produce simple background histograms from the pandas dataframe
  * ```readDF_04.py```: compare signal to backgrounds, with the proper normalisations,
    with the backgrounds stacked on top of each other     
  * ```readDF_05.py```: compare signal to backgrounds, in shapes
  * ```testDNN_01.py```: operate standard transformations on input variables
    when using a single background sample (works with ```params-boost_sig_mu_01.cfg```)
  * ```testDNN_02.py```: prepare one input signal and one input bkg sample, 
    build a DNN model based on parameters from the config file (works with ```params-boost_sig_mu_01.cfg```)
  * ```testDNN_03.py```: configure the first training of the DNN model (works with ```params-boost_sig_mu_01.cfg```)
  * ```testDNN_04.py```: add dropout layers (works with ```params-boost_sig_mu_02.cfg```)
  * ```testDNN_05.py```: check dropout layers with toys (works with ```params-boost_sig_mu_02.cfg```)
  * ```testDNN_06.py```: fix the Adam optimizer and implement some more input params (works with ```params-boost_sig_mu_02.cfg```)
  * ```readHistory.py```: example on ho to read a saved DNN training history file (works with ```params-boost_sig_mu.cfg```)
  * ```plotTrainingTrends.py```: use the output of ```testDNN_05.py``` to look at training figure of merit trends
  # ```checkWeights.py```: simple training model to see the effect of event weights


## TODO

  * plotting
    * get the right inclusive cross-sections and efficiencies --> from Davide or Giacomo
    * add variables calculated from the existing ones
      * e.g. mjj VBS
      * various etas
      * various zeppenfeld's
    * fix the edgecolors problem in the stack plots  
    * add overflow and underflow bins 
  * variables
    * Wasserstein distance
    * SHAP
  * DNN
    * first training with signal and one background  
    * early stopping
    * learning updating e learning rate (dipende dal numero di epoche prestabilite?), 
      che metodo di minimizzazione sto utilizzando?
    * weights decay
  * codice
    * training output folder with all needed infoes and plots
    * check mem leaks in testDNN_05

## QUESTIONS

  * why does the DY in the boosted_mu category have events with weight 0?
  * why, with no regularisations, the validation distributions fluctuate much more than the train ones?
    * with 50-50 splitting of the samples
    * it remains the same also if the samples are swapped
  * why isn't the result of the training deterministic?
  * why is the overtraining visibile only in the loss function and non in other metrics?
    * AUC shows it as well somehow
  * how do weights are used? (see discussion w/ Simone and Giulia)

## USEFUL LINKS

  * [MIB code example](https://github.com/UniMiBAnalyses/ML_classification/blob/master/Training_v3_resolved_weights_Aurora.ipynb)
  * [Adam optimizer](https://arxiv.org/abs/1412.6980)
  * [program creek[(https://www.programcreek.com/)
