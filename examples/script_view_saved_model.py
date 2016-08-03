from python_gclda_package.gclda_dataset import gclda_dataset
from python_gclda_package.gclda_model   import gclda_model
import cPickle as pickle
import gzip

# ------------------------------------------------------
# --- Configure gcLDA model and dataset Parameters   ---
# ------------------------------------------------------

datasetLabel = '2015Filtered2_1000docs'

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

# -------------------------------------------------
# --- Load Model From File and Display Summary  ---
# -------------------------------------------------

# Set up model filename to load
results_rootdir = 'gclda_results'
results_outputdir = '%s/%s' % (results_rootdir, model_str)
results_modelfile = '%s/results_iter%02d.p' % (results_outputdir, current_iter)

# Load compressed model object
print 'loading model'
with gzip.open(results_modelfile,'rb') as f:
	model = pickle.load(f)
model.displayModelSummary()

