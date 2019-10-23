#######################
# Author: Will Spaeth #
# Date: 9-10-19       #
#######################

#####################
# Philosophy of Use #
#####################
# Packages required: listed in Dev-Analyzer-Env.yaml
#
# This program acts as a sort of wrapper for the entire model training and evaluation process.
# It's still in a demo mode and geared specifically for a certain project and model (kinematic data analysis and AnalysisNetwork),
# but with some more work and modifications it could accommodate roughly any standard model, datset, training regimen, and
# evaluation process.
#
# When running a new model, the only new code development that should need to occur are in the following files:
# 	• lib/classes/model_classes: create your new model class here, copying the conventions in the current AnalysisNetwork class
# 	• lib/classees/data_classes: create your new data class here, copying the conventions in the current SubjectDataset class
#
# Additionally, the configuration files should be modified to load your dataset, model, and training / evaluation regimen. By
# default, the program will always load the default.py configuration files located in each config folder, and merge them to create
# a master configuration file, which gets overwritten by the search or evaluation parameters you wish to run. Different default
# configuration files can be declared as command line arguments, as detailed in the lib.control_loop.py file. To determine which
# experiments/evaluations get performed, edit the combination configuration file with your parameter lists.
#
#
# To recap, the process is:
# 	(1) Create new model and dataset classes in the proper style
#   (2) If necessary, edit or create new configuration default files
#   (3) If necessary, edit or create combination configuration file to create all experiments/evaluations
#   (4) Run the main.py method with the desired arguments, with options listed in lib/control_loop.py
# 		(4.1) If you don't want to type arguments every time, create a wrapper script for the main.py script that feeds
# 				the proper arguments

################################################################
# Third Party Packages Manually Installed Through Conda 4.7.11 #
################################################################
# • python=3.7.4
# • numpy=1.16.4
# • pandas=0.25.0
# • matplotlib=3.1.0
# • yaml=0.1.7
# • yacs=0.1.6
# • keras=2.2.4
# • nomkl=3.0

####################
# Testing Schedule #
####################
# • Test multiple filters with sparsity
# 	• Use single set of features, with two filters
# 	• Use multiple sets of features, with two filters
# 	• Use single set of features, with three filters
# 	• Use multiple sets of features, with three filters
# • Test on velocity data
# • Test on multiple infant data
# 	• Properly set temporal sample weighting for every infant
# 	• Save filter response figures for every sample
# • Create model to copy convolutional filter and find the proper offset
# 	• Test on infants with which the original filter was trained
# 	• Test on infants with cerebral palsy

######################
# Development Status #
######################
### Short term
# • Implement inner product penalty
# 	• Frobenius inner product
#	• Regularization loss needs to be separable from other losses
# 	• Needs to apply between every combination of filters
# • Put loss functions on the dual graphs
# • Remove plot for nan values on both label plots
# • More visualization graphs
### Longterm
# • Need to implement reload function, for training or evaluation
# • Need to generalize the methods to accommodate any static model within keras
# 	• Probably need to feed master config to all relevant methods, since different models and training regiments may need wildly different configuration parameter access
#	• Generalize data loading to accommodate any dataset
# • Maybe convert fully to model subclassing
# 	• This means modifying the model class, trainer method, checkpointing mechanism, and saving mechanism

#########################
# Experimental Branches #
#########################
# • Better sparsity
# • More than one filter with orthogonality regularization
# • Better visualization
# • Attempt normal experiments on velocity alone

