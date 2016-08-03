# This script illustrates how to evaluate the predicted loglikelihood of a holdout dataset using a trained gclda model

from python_gclda_package.gclda_dataset import gclda_dataset
from python_gclda_package.gclda_model   import gclda_model
import cPickle as pickle
import gzip
import numpy as np


# ------------------------------------------------------
# --- Configure gcLDA model and dataset Parameters   ---
# ------------------------------------------------------

datasetLabel = '2015Filtered2_TrnTst1P1'

current_iter = 1000 # Saved model iteration

nt  		= 100	# Number of topics
nr 			= 2 	# Number of subregions (any positive integer, but must equal 2 if symmetric subregion model)
alpha 		= .1 	# Prior count on topics for each doc
beta 		= .01 	# Prior count on word-types for each topic
gamma 		= .01 	# Prior count added to y-counts when sampling z assignments
delta 		= 1.0 	# Prior count on subregions for each topic
roi 		= 50 	# Default spatial 'Region of interest' size (default value of diagonals in covariance matrix for spatial distribution, which the distributions are biased towards)
dobs 		= 25 	# Region 'default observations' (# pseudo-observations biasing Sigma estimates in direction of default 'roi' value)
symmetric 	= True	# Use symmetry constraint on subregions? (symmetry requires nr = 2)

seed_init 	= 1 	# Initial value of random seed

# --- Set up model_str identifier for saving/loading results based on model params ---
model_str = '%s_%dT_%dR_alpha%.3f_beta%.3f_gamma%.3f_delta%.3f_%ddobs_%.1froi_%dsymmetric_%d' % (datasetLabel,
	nt, nr, alpha, beta, gamma, delta, dobs, roi, symmetric, seed_init)


# ----------------------------------------------
# --- Create dataset object for holdout-data ---
# ----------------------------------------------

# Assumes that the document-indices are aligned between train/test datasets 

# Create a dataset object for test-data
test_datasetLabel = '2015Filtered2_TrnTst1P2'
dataDirectory = '../datasets/neurosynth/'

# Create dataset object & Import data
testdat = gclda_dataset(test_datasetLabel,dataDirectory)
testdat.importAllData()

# --------------------------------
# --- Load trained gcLDA model ---
# --------------------------------

# Set up model filename to load
results_rootdir = 'gclda_results'
results_outputdir = '%s/%s' % (results_rootdir, model_str)
results_modelfile = '%s/results_iter%02d.p' % (results_outputdir, current_iter)

# Load compressed model object
print 'loading model'
with gzip.open(results_modelfile,'rb') as f:
	model = pickle.load(f)

# -----------------------------------------------------------------
# --- Run Log-likely computation on both training and test-data ---
# -----------------------------------------------------------------

loglikelyout_train = model.computeLogLikelihood(model.dat, False) # Recompute gcLDA log-likelihood on training data
loglikelyout_test = model.computeLogLikelihood(testdat, False) # Compute trained gcLDA log-likelihood on holdout data


# Print log-likelihoods for train and test:
print 'log-likely train:'
print 'loglikely_x | loglikely_w | loglikely_tot'
print '%5.1f  %5.1f  %5.1f' % (loglikelyout_train[0], loglikelyout_train[1], loglikelyout_train[2])

print 'log-likely test:'
print 'loglikely_x | loglikely_w | loglikely_tot'
print '%5.1f  %5.1f  %5.1f' % (loglikelyout_test[0], loglikelyout_test[1], loglikelyout_test[2])

