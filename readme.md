statsrat is a Python package for using learning models in psychological research.
It is currently intended for use with category learning and Pavlovian conditioning, but 
in the future I hope to expand it to deal with other types of experiment.
The package's core functionality includes the following:
-easily create new learning models using modular code
-easily create new experimental designs using a simple description
-fit learning models to time step level individual data
    -import experimental data and automatically prepare for model fitting
    -fit by individual maximum likelihood or the EM algorithm or MCMC
    -perform recovery tests using simulated data
    -perform cross-validation by fitting to one part of a data set and testing prediction on the rest
-fit models to group level data
    -ordinal adequacy tests (OATs)
    -learning curves
    -test probabilities
-examine model performance through graphs of association weights, attention weights, prediction error etc.

author: Samuel Paskewitz
