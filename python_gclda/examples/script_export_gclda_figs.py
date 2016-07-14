# This script loads a saved gclda model, and exports figures illustratings topics 
# (and some additional model details) to file

from python_gclda_package.gclda_dataset import gclda_dataset
from python_gclda_package.gclda_model   import gclda_model
import cPickle as pickle
import gzip

# ------------------------------------------------------
# --- Configure gcLDA model and dataset Parameters   ---
# ------------------------------------------------------

datasetLabel = '2015Filtered2_TrnTst1p1'

current_iter = 5 # Saved model iteration

nt  		= 100	# Number of topics
nr 			= 2 	# Number of subregions (should work with any number)
alpha 		= .1 	# Prior count on topics for each doc
beta 		= .01 	# Prior count on word-types for each topic
gamma 		= .01 	# Prior count added to y-counts when sampling z assignments
delta 		= 1.0 	# Prior count on subregions for each topic
roi 		= 50 	# Default ROI (default covariance spatial region we regularize towards)
dobs 		= 25 	# Sample constant (# observations weighting sigma in direction of default covariance)
symmetric 	= True	# Use symmetry constraint on subregions? (symmetry requires nr = 2)

seed_init 	= 1 	# Initial value of random seed

# --- Set up model_str identifier for saving/loading results based on model params ---
model_str = '%s_%dT_%dR_alpha%.3f_beta%.3f_gamma%.3f_delta%.3f_%ddobs_%.1froi_%dsymmetric_%d' % (datasetLabel,
	nt, nr, alpha, beta, gamma, delta, dobs, roi, symmetric, seed_init)

# -------------------------------------------
# --- Export model params and Print Figs  ---
# -------------------------------------------

# Set up model filename to load
results_rootdir = 'gclda_results'
results_outputdir = '%s/%s' % (results_rootdir, model_str)
results_modelfile = '%s/results_iter%02d.p' % (results_outputdir, current_iter)

# Load compressed model object
print 'loading model'
with gzip.open(results_modelfile,'rb') as f:
	model = pickle.load(f)
model.displayModelSummary()

# Configure the output directories for the current results (specifying the iteration) and print
print 'exporting model params'
outputdir = '%s/iter_%02d/' % (results_outputdir, model.iter)
model.printAllModelParams(outputdir) 	# Prints model parameters to files
print 'printing figures'
outputdir = '%s/iter_%02d/figs' % (results_outputdir, model.iter)
model.printTopicFigures(outputdir)  	# Print topic figures to png files


