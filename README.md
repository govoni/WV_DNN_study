# DNN for QML studies

## content of the folder

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

## TODO

  * plotting
    * get the right inclusive cross-sections
    * add variables calculated from the existing ones
      * e.g. mjj VBS
      * various etas
      * various zeppenfeld's
    * fix the edgecolors problem in the stack plots  
    * add overflow and underflow bins 
    * plot a comparison of signal to bkgs's in shapes
  * variables
    * preprocess variables
      * same ranges (apply same preprocessing to signal and bkg)
      * same number of events for each sample used in the training
    * wasserstein distance
    * SHAP
  * DNN
    * first training with signal and one background  

## QUESTIONS

  * why does the DY in the boosted_mu category have events with weight 0?