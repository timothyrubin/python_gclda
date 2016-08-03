# This file illustrates basic usage of the python_gclda toolbox. Specifically, we do the following steps:
#	1 - Import the python modules
# 	2 - Build a python dataset object, and import data into the object from raw text files containing all data that the gcLDA model uses
#	3 - Build a gclda model object instance
#	4 - Train the gclda model object (using fewer iterations than should be used for actual modeling)
#	5 - Export figures 
#
# Note: this tutorial will assume that your working directory is the '/examples' subdirectory within
#	the gclda package. If it is not, the relative paths to the datasets need to be modified for 
#	creating the variable 'datasetLabel' below


# Import packages
from python_gclda_package.gclda_dataset import gclda_dataset
from python_gclda_package.gclda_model   import gclda_model
import os

# -----------------------------------------------
# --- Create a dataset object and import data ---
# -----------------------------------------------

# For this tutorial, we use a subset of 1000 documents from the neurosynth dataset. 
# This will run faster, but produce sparser and noisier topics than the full dataset

# Create dataset object instance: 'dat'
# Inputs:
datasetLabel  ='2015Filtered2_1000docs'				# The directory name containing the dataset .txt files, which will be used as a 'dataset label'
datasetDirectory =	'../datasets/neurosynth/' 		# The relative path from the working directory to the root-directory containing the dataset folder
dat = gclda_dataset(datasetLabel,datasetDirectory) 	# Create dataset object 'dat'
# Import data from all files that are in dataset directory:
dat.importAllData()
# View dataset object after importing data:
dat.displayDatasetSummary()

# -----------------------------------
# --- Create a gclda model object ---
# -----------------------------------

# Create gclda model, using T=100 topics, R=2 subregions per topic, and default values for all other hyper-parameters
# 	(See other sample scripts for details about all hyperparameters)
T = 100
R = 2

model = gclda_model(dat, T, R)
# Initialize the model
model.initialize()
# View the model after initialization:
model.displayModelSummary()

# ---------------------------------------------------
# --- Run the gcLDA training for a few iterations ---
# ---------------------------------------------------

# Note that we run for just a few iterations for the sake of time here.
#	 When training a model, we recommend running for at least 1000 total iterations
iterations = 25
# During training, the model will print details about the model log-likelihood, etc., to the console. Verbosity arguments can change that.
for i in range(iterations):
	model.runCompleteIteration()


# ---------------------------------------------
# --- Export figures illustrating the model ---
# ---------------------------------------------

# Set up a rootdirectory to serve as a directory to store all tutorial results
results_rootdir = 'gclda_tutorial_results' # We note that these results come from the tutorial, as opposed to the scripts for running full models
if not os.path.isdir(results_rootdir):
	os.mkdir(results_rootdir)

# We first use a method to get a string identifier that is unique to the combination of:
#		- DatasetLabel
#		- Model hyperparameters
# This is useful for saving model output

# Get model string identifier to use as a results directory
modelString = model.getModelDisplayString()
# Append the current model iteartion to this directory name
outputDirectory_data = results_rootdir + "/" + modelString + "_Iteration_%d/" % model.iter
# We create a subdirectory of the outputdirectory to store figures
outputDirectory_figures = outputDirectory_data + "Figures/"

# Export some files giving topic-word distributions, as well as detailed accounts of all token->assignments
model.printAllModelParams(outputDirectory_data)
# Export Figures illustrating the spatial-extent and high probability word-types for the each topic in the model
model.printTopicFigures(outputDirectory_figures, 1)




