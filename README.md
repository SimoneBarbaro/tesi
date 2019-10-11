# Analisi di tecniche di preprocessing per la classificazione di coralli con deep learning
Implementation of the project for the thesis: Barbaro, Simone (2019) Analisi di tecniche di preprocessing per la classificazione di coralli con deep learning. [Laurea], Università di Bologna, Corso di Studio in Ingegneria e scienze informatiche [LDM270] - Cesena. http://amslaurea.unibo.it/18481/.
The program can run all the experiments listed on the thesis and can be used as described below.
## Usage
src/main.py [config_name [res_dir [save_testing]]]
config_name: name of the config file to be used in the run to specify preprocessing, model and hyperparameter search. There must be a file called config_name + "_config.json" in the folder config. These files are the configurations of the experiments for this thesis. If not specified it will be used a testing configuration.
res_dir: directory where to store run results. These include confusion matrices, trained models, logs and validation performances. If not specified it will be used the standard output for printing validation performances and the rest of the data will not be saved.
save_testing: if True, the models will be tested on the test set and the results will be stored in res_dir.
