import numpy as np
from scipy.stats import multivariate_normal
import os
import matplotlib.pyplot as plt

# Class object for a gcLDA dataset
class gclda_model:
	# -------------------------------------------------------
	# <<<<< Model Constructor and Initialization Method >>>>>
	# -------------------------------------------------------

	# -----------------------------------------------------------------------
	#  Constructor: Set up all object variables using gcLDA parameter values
	# -----------------------------------------------------------------------
	def __init__(self, dat, nt = 100, nr = 2, alpha = .1, beta = .01, gamma = .01, delta = 1.0, dobs=25.0, roi=50.0, symmetric = False, seed_init = 1):
		'''
		Constructor: Create a gcLDA model using a dataset object and hyperparameter arguments

		Input parameters:
			dat 		Dataset object
			nt		 	Number of topics
			nr 		 	Number of subregions (>=1)
			alpha 		Prior count on topics for each doc
			beta 		Prior count on word-types for each topic
			gamma 		Prior count added to y-counts when sampling z assignments
			delta 		Prior count on subregions for each topic
			roi 		Default spatial 'Region of interest' size (default value of diagonals in covariance matrix for spatial distribution, which the distributions are biased towards)
			dobs 		Spatial Region 'default observations' (# observations weighting Sigma estimates in direction of default 'roi' value)
			symmetric 	Use symmetry constraint on subregions? (symmetry requires nr = 2)
			seed_init 	Initial value of random seed
		'''

		print 'Constructing GC-LDA Model'
		# --- Checking to make sure parameters are valid
		if symmetric==True:
			if not nr==2:
				raise ValueError('Cannot run a symmetric model unless #Subregions (nr) == 2 !') # symmetric model only valid if R = 2

		# --- Assign dataset object to model
		self.dat = dat

		# --- Initialize sampling parameters
		self.iter = 0 					# Tracks the global sampling iteration of the model 
		self.seed_init = seed_init 		# Random seed for initializing model
		self.seed = 0 					# Tracks current random seed to use (gets incremented after initialization and each sampling update)

		# --- Set up gcLDA model hyper-parameters from input
		# Pseudo-count hyperparams need to be floats so that when sampling distributions are computed the count matrices/vectors are converted to floats
		self.nt = nt 					# Number of topics (T)
		self.nr = nr 					# Number of subregions (R)
		self.alpha 	= float(alpha)  	# Prior count on topics for each doc (\alpha)
		self.beta 	= float(beta) 		# Prior count on word-types for each topic (\beta)
		self.gamma 	= float(gamma) 		# Prior count added to y-counts when sampling z assignments (\gamma)
		self.delta 	= float(delta)		# Prior count on subregions for each topic (\delta)
		self.roi 	= float(roi)		# Default ROI (default covariance spatial region we regularize towards) (not in paper)
		self.dobs 	= float(dobs)		# Sample constant (# observations weighting sigma in direction of default covariance) (not in paper)
		self.symmetric = symmetric		# Use constrained symmetry on subregions? (only for nr = 2)

		# --- Get dimensionalities of vectors/matrices from dataset object
		self.nz = len(dat.widx) 		# Number of word-tokens
		self.ny = len(dat.peak_didx) 	# Number of peak-tokens
		self.nw = len(dat.wlabels) 		# Number of word-types
		self.nd = max(dat.wdidx) 		# Number of documents
		self.nxdims = dat.peak_ndims	# Dimensionality of peak_locs data
		
		#  --- Preallocate vectors of Assignment indices
		self.zidx = np.zeros(self.nz, dtype = int) # word->topic assignments (z)
		self.yidx = np.zeros(self.ny, dtype = int) # peak->topic assignments (y)
		self.ridx = np.zeros(self.ny, dtype = int) # peak->subregion assignments (c)
		
		#  --- Preallocate count matrices
		self.ny_d_t 	= np.zeros(shape = (self.nd, self.nt), dtype = int) # Peaks: D x T: Number of peak-tokens assigned to each topic per document
		self.ny_r_t 	= np.zeros(shape = (self.nr, self.nt), dtype = int) # Peaks: R x T: Number of peak-tokens assigned to each subregion per topic
		self.nz_w_t 	= np.zeros(shape = (self.nw, self.nt), dtype = int) # Words: W x T: Number of word-tokens assigned to each topic per word-type
		self.nz_d_t 	= np.zeros(shape = (self.nd, self.nt), dtype = int) # Words: D x T: Number of word-tokens assigned to each topic per document 
		self.nz_sum_t 	= np.zeros(shape = (1,self.nt), dtype = int)      	# Words: 1 x T: Total number of word-tokens assigned to each topic (across all docs) 
		
		#  --- Preallocate Gaussians for all subregions
		#  Regions_Mu & Regions_Sigma: Gaussian mean and covariance for all subregions of all topics
		#  Formed using lists (over topics) of lists (over subregions) of numpy arrays 
		#		regions_mu    = (nt, nr,      1, nxdims) 
		#		regions_sigma = (nt, nr, nxdims, nxdims)
		self.regions_mu = []
		self.regions_sigma = []
		for t in xrange(self.nt):
			topic_mu =[]
			topic_sigma = []
			for r in xrange(self.nr):
				topic_mu.append( np.zeros(shape = (1, self.nxdims)) )  
				topic_sigma.append( np.zeros(shape = (self.nxdims, self.nxdims)) )
			self.regions_mu.append(topic_mu) 		# (\mu^{(t)}_r)
			self.regions_sigma.append(topic_sigma)  # (\sigma^{(t)}_r)

		#  --- Initialize lists for tracking log-likelihood of data over sampling iterations
		self.loglikely_iter = []	# Tracks iteration we compute each loglikelihood at
		self.loglikely_x 	= []	# Tracks log-likelihood of peak tokens
		self.loglikely_w 	= [] 	# Tracks log-likelihood of word tokens
		self.loglikely_tot 	= [] 	# Tracks log-likelihood of peak + word tokens
	
	# ------------------------------------------------------------------------------------
	#  Random Initialization: Initial z, y, r assignments. Get Initial Spatial Estimates
	# ------------------------------------------------------------------------------------
	def initialize(self):
		print 'Initializing GC-LDA Model'

		# --- Seed random number generator
		np.random.seed(self.seed_init)

		# --- Randomly initialize peak->topic assignments (y) ~ unif(1...nt)
		self.yidx[:] = np.random.randint( self.nt , size = (self.ny))
		# --- Initialize peak->subregion assignments (r) 
		#		if asymmetric model, randomly sample r ~ unif(1...nr)
		#		if symmetric model use deterministic assignment : if peak_val[0] > 0, r = 1, else r = 0
		if not self.symmetric:
			self.ridx[:] = np.random.randint( self.nr , size = (self.ny))
		else:
			self.ridx[:] = (self.dat.peak_vals[:,0]>0).astype(int)

		# Update model vectors and count matrices to reflect y and r assignments
		for i in xrange(self.ny):
			d = self.dat.peak_didx[i]-1 # document -idx (d): we use "-1" to convert from 1-based indexing (in data) to 0-based indexing (in python)
			y = self.yidx[i] 			# peak-token -> topic assignment (y_i)
			r = self.ridx[i] 			# peak-token -> subregion assignment (c_i)
			self.ny_d_t[d,y] += 1 		# Increment document-by-topic counts
			self.ny_r_t[r,y] += 1 		# Increment region-by-topic

		# --- Randomly Initialize Word->Topic Assignments (z) for each word token w_i: sample z_i proportional to p(topic|doc_i)
		for i in xrange(self.nz):
			w = self.dat.widx[i]-1 		# w_i word-type: convert widx from 1-based to 0-based indexing 
			d = self.dat.wdidx[i]-1 	# w_i doc-index: convert wdidx from 1-based to 0-based indexing
			
			# Estimate p(t|d) for current doc
			p_t_d = self.ny_d_t[d] + self.gamma
			
			# Sample a topic from p(t|d) for the z-assignment
			probs = np.cumsum(p_t_d) 						# Compute a cdf of the sampling distribution for z
			sample_locs = probs<np.random.rand()*probs[-1] 	# Which elements of cdf are less than random sample?
			sample_locs = np.where(sample_locs) 			# How many elements of cdf are less than sample
			z = len(sample_locs[0]) 						# z = # elements of cdf less than rand-sample
			
			# Update model assignment vectors and count-matrices to reflect z
			self.zidx[i] = z 			# Word-token -> topic assignment (z_i)
			self.nz_w_t[w,z] += 1
			self.nz_sum_t[0,z] += 1
			self.nz_d_t[d,z] +=1

		# --- Get Initial Spatial Parameter Estimates 
		self.update_regions()

		# --- Get Log-Likelihood of data for Initialized model and save to variables tracking loglikely
		self.computeLogLikelihood(self.dat)
	
	# -------------------------------------------------------------------------------
	# <<<<< Model Parameter Update Methods >>>> Update z, Update y/r, Update regions  |
	# -------------------------------------------------------------------------------

	# -------------------------------------------------------------------
	#  Run a complete update cycle (sample z, sample y&r, update regions)
	# -------------------------------------------------------------------
	# Verbosity argument determines how much info we print to console
	def runCompleteIteration(self, loglikely_Freq = 1, verbose = 2):
		self.iter += 1 				# Update total iteration count
		if verbose>1:
			print 'iter %02d sampling z' % self.iter
		self.seed += 1
		self.update_z(self.seed) 	# Update z-assignments
		if verbose>1:
			print 'iter %02d sampling y|r' % self.iter
		self.seed += 1
		self.update_y(self.seed) 	# Update y-assignments
		if verbose >1:
			print 'iter %02d updating spatial params' % self.iter
		self.update_regions() 		# Update gaussian estimates for all subregions
		# Only update loglikelihood every 'loglikely_Freq' iterations (Computing log-likelihood isn't necessary and slows things down a bit)
		if (self.iter % loglikely_Freq == 0 ):
			if verbose >1:
				print 'iter %02d computing log-likelihood' % self.iter
			self.computeLogLikelihood(self.dat) # Compute log-likelihood of model in current state
			if verbose >0:
				print 'Iter %02d Log-likely: x = %10.1f, w = %10.1f, tot = %10.1f' % (self.iter, self.loglikely_x[-1], self.loglikely_w[-1], self.loglikely_tot[-1])

	# -------------------------------------------------------------------
	#  Update z indicator variables assigning words->topics
	# -------------------------------------------------------------------
	def update_z(self, randseed):
		# --- Seed random number generator
		np.random.seed(randseed)

		# Loop over all word tokens
		for i in range(self.nz):
			# Get indices for current token
			w =  self.dat.widx[i] - 1  	# w_i word-type: convert widx from 1-based to 0-based indexing 
			d =  self.dat.wdidx[i] - 1 	# w_i doc-index: convert wdidx from 1-based to 0-based indexing
			z = self.zidx[i]			# current Topic-assignment for word token w_i

			# Decrement count-matrices to remove current zidx
			self.nz_w_t[w,z]   -=1
			self.nz_sum_t[0,z] -=1
			self.nz_d_t[d,z]   -=1

			# Get Sampling distribution:
			#	p(z_i|z,d,w) ~ p(w|t) * p(t|d)
			#				 ~ p_w_t * p_t_d
			p_w_t = (self.nz_w_t[w,:] + self.beta) / (self.nz_sum_t +  self.beta * self.nw)
			p_t_d = self.ny_d_t[d,:] + self.gamma
			probs = p_w_t * p_t_d # The unnormalized sampling distribution

			# Sample a z_i assignment for the current word-token from the sampling distribution
			probs = np.squeeze(probs) / np.sum(probs) 	# Normalize the sampling distribution
			vec = np.random.multinomial(1, probs) 		# Numpy returns a [1 x T] vector with a '1' in the index of sampled topic 
			vec_loc = np.where(vec)						# Transform the indicator vector into a single z-index (stored in tuple)
			z = vec_loc[0][0]							# Extract the sampled z value from the tuple

			# Update the indices and the count matrices using the sampled z assignment
			self.zidx[i] = z 					# Update w_i topic-assignment
			self.nz_w_t[w,z] 	+=1
			self.nz_sum_t[0,z] 	+=1
			self.nz_d_t[d,z] 	+=1

	# ---------------------------------------------------------------------
	#  Update y / r indicator variables assigning peaks->topics/subregions
	# ---------------------------------------------------------------------
	def update_y(self, randseed):
		# --- Seed random number generator
		np.random.seed(randseed)

		# Retrieve p(x|r,y) for all subregions
		peak_probs = self.getPeakProbs(self.dat)

		# eyemat: matrix that is added to the current doc_y_counts to generate the 'proposed' doc_y_counts. Precomputed for efficiency
		#eyemat = np.eye(self.nt)

		# Iterate over all peaks x, and sample a new y and r assignment for each
		for i in range(self.ny):

			d = self.dat.peak_didx[i]-1
			y = self.yidx[i]
			r = self.ridx[i]

			self.ny_r_t[r,y] -= 1 # Decrement count in Subregion x Topic count matrix
			self.ny_d_t[d,y] -= 1 # Decrement count in Document x Topic count matrix

			# Retrieve the probability of generating current x from all subregions: [R x T] array of probs
			p_x_subregions = (peak_probs[i,:,:]).transpose()

			# --- Compute the probabilities of all subregions given doc: p(r|d) ~ p(r|t) * p(t|d) ---
			p_r_t = self.ny_r_t + self.delta     	# Counts of subregions per topic + prior: p(r|t)
			p_r_t = p_r_t / np.sum(p_r_t,axis=0) 	# Normalize the columns such that each topic's distribution over subregions sums to 1
			p_t_d = self.ny_d_t[d,:] + self.alpha 	# Counts of topics per document + prior: p(t|d) 
			# Compute p(subregion | document): p(r|d) ~ p(r|t) * p(t|d)
			p_r_d = np.ones([self.nr , 1]) * p_t_d * p_r_t 			# [R x T] array of probs

			# --- Compute the multinomial probability: p(z|y) ---
			# Need the current vector of all z and y assignments for current doc
			doc_y_counts = self.ny_d_t[d,:] + self.gamma # The multinomial from which z is sampled is proportional to number of y assigned to each topic, plus constant \gamma
			doc_z_counts = self.nz_d_t[d,:]
			p_z_y = np.zeros([1,self.nt])
			p_z_y[:] = self.compute_PropMultinomial_From_zy_Vectors(doc_z_counts,doc_y_counts)

			# Compute probability of observing word->topic assignments (z) given the vectors for all proposed peak->topic assignments (y): p(z|y)
			# proposed_y_counts = np.dot(np.ones([self.nt,1]),doc_y_counts.reshape([1,len(doc_y_counts)]))
			# proposed_y_counts += eyemat # Add eyemat to convert a matrix of current doc_y_counts to a matrix of proposed doc_y_counts
			# p_z_y[:] = self.mnpdf_proportional(doc_z_counts, proposed_y_counts) # Returns a vector proportional to p(z_d|y_d) 

			# === Block sampling c_i and y_i assignments =====
			# Now Compute the full sampling distribution: p(y_i,c_i|y,c,x,r,d) ~ p(x|mu, sigma) * p(r|d) * multinomial p(z|y)
			# For subregions this decomposes into: p(y_i = t, c_i = r|y,x,r,d) ~ p(x|mu_r, sigma_r) * p(r|t) * p(t|d) * multinomial p(z|y)

			# Get the full sampling distribution: 
			probs_pdf = p_x_subregions * p_r_d * np.dot( np.ones([self.nr, 1]), p_z_y ) # [R x T] array containing the proportional probability of all y/r combinations
			probs_pdf = probs_pdf.transpose().ravel() 	# Convert from a [R x T] matrix into a [R*T x 1] array we can sample from
			probs_pdf = probs_pdf / np.sum(probs_pdf) 	# Normalize the sampling distribution
			
			# Sample a single element (corresponding to a y_i and c_i assignment for the peak token) from the sampling distribution
			vec = np.random.multinomial(1, probs_pdf) 	# Returns a [1 x R*T] vector with a '1' in location that was sampled
			vec_loc = np.where(vec) 					# Converts the indicator vector into a linear index value (stored in a tuple)
			sample_idx = vec_loc[0][0]					# Extract the linear index value from the tuple

			# Transform the linear index of the sampled element into the subregion/topic (r/y) assignment indices
			r = np.remainder(sample_idx, self.nr)   # Subregion sampled (r)
			y = sample_idx / self.nr 				# Topic sampled (y)

			# Update the indices and the count matrices using the sampled y/r assignments
			self.ny_r_t[r,y] += 1 	# Increment count in Subregion x Topic count matrix
			self.ny_d_t[d,y] += 1 	# Increment count in Document x Topic count matrix
			self.yidx[i] = y 		# Update y->topic assignment
			self.ridx[i] = r 		# Update y->subregion assignment

	# ------------------------------------------------------------------------------
	#  Update spatial distribution parameters (Gaussians params for all subregions)
	# ------------------------------------------------------------------------------
	def update_regions(self):
		# Generate default ROI based on default_width
		A = self.roi * np.eye(self.nxdims)

		if not self.symmetric:
			# --- If model subregions not symmetric ---
			# For each region, compute a mean and a regularized covariance matrix
			for t in range(self.nt):
				for r in range(self.nr):
					
					# -- Get all peaks assigned to current topic & subregion --
					idx = (self.yidx == t) & (self.ridx==r)
					vals = self.dat.peak_vals[idx]
					nobs = self.ny_r_t[r,t] 
					
					# -- Estimate Mean --
					# If there are no observations, we set mean equal to zeros, otherwise take MLE
					if nobs == 0:
						mu = np.zeros([self.nxdims])
					else:
						mu = np.mean(vals,axis=0)
					
					# -- Estimate Covariance --
					# if there are 1 or fewer observations, we set sigma_hat equal to default ROI, otherwise take MLE
					if nobs<=1:
						C_hat = A
					else:
						C_hat = np.cov(np.transpose(vals))

					# Regularize the covariance, using the ratio of observations to dobs (default constant # observations)
					d_c = nobs / ( nobs + self.dobs)
					sigma = d_c * C_hat + (1-d_c) * A;
					
					# --  Store estimates in model object --
					self.regions_mu[t][r][:] = mu
					self.regions_sigma[t][r][:] = sigma
		else:
			# --- If model subregions are symmetric ---
			# With symmetric subregions, we jointly compute all estimates for subregions 1 & 2, constraining the means to be symmetric w.r.t. the origin along x-dimension
			for t in range(self.nt):
					
				# -- Get all peaks assigned to current topic & subregion 1 --
				idx1 = (self.yidx == t) & (self.ridx==0)
				vals1 = self.dat.peak_vals[idx1]
				nobs1 = self.ny_r_t[0,t]

				# -- Get all peaks assigned to current topic & subregion 2 --
				idx2 = (self.yidx == t) & (self.ridx==1)
				vals2 = self.dat.peak_vals[idx2]
				nobs2 = self.ny_r_t[1,t]

				# -- Get all peaks assigned to current topic & either subregion --
				allvals = self.dat.peak_vals[idx1|idx2]

				# --------------------
				# -- Estimate Means --
				# --------------------
				
				# -- Estimate Independent Mean For Subregion 1 --
				# If there are no observations, we set mean equal to zeros, otherwise take MLE
				if nobs1 == 0:
					m = np.zeros([self.nxdims])
				else:
					m = np.mean(vals1,axis=0)
				
				# -- Estimate Independent Mean For Subregion 2 --
				# If there are no observations, we set mean equal to zeros, otherwise take MLE
				if nobs2 == 0:
					n = np.zeros([self.nxdims])
				else:
					n = np.mean(vals2,axis=0)

				# -- Estimate the Weighed means of all dims, where for dim1 we compute the mean w.r.t. absolute distance from the origin
				weighted_mean_dim1 = ( -m[0]*nobs1 + n[0]*nobs2 ) / (nobs1 + nobs2) 
				weighted_mean_otherdims = np.mean(allvals[:,1:],axis=0)

				# Store weighted mean estimates 
				mu1 = np.zeros([1,self.nxdims])
				mu2 = np.zeros([1,self.nxdims])
				mu1[0,0]  = -weighted_mean_dim1
				mu1[0,1:] = weighted_mean_otherdims
				mu2[0,0]  = weighted_mean_dim1
				mu2[0,1:] = weighted_mean_otherdims

				# --  Store estimates in model object --
				self.regions_mu[t][0][:] = mu1
				self.regions_mu[t][1][:] = mu2

				# --------------------------
				# -- Estimate Covariances --
				# --------------------------

				# Covariances are estimated independently
				# Cov for subregion 1
				if nobs1<=1:
					C_hat1 = A
				else:
					C_hat1 = np.cov(np.transpose(vals1))
				# Cov for subregion 2
				if nobs2<=1:
					C_hat2 = A
				else:
					C_hat2 = np.cov(np.transpose(vals2))

				# Regularize the covariances, using the ratio of observations to sample_constant
				d_c_1 = (nobs1 ) / ( nobs1 + self.dobs)
				d_c_2 = (nobs2 ) / ( nobs2 + self.dobs)
				sigma1 = d_c_1 * C_hat1 + (1-d_c_1) * A;
				sigma2 = d_c_2 * C_hat2 + (1-d_c_2) * A;

				# --  Store estimates in model object --
				self.regions_sigma[t][0][:] = sigma1
				self.regions_sigma[t][1][:] = sigma2

	# --------------------------------------------------------------------------------
	# <<<<< Utility Methods for GC-LDA >>>>> Log-Likelihood, Get Peak-Probs , mnpdf  |
	# --------------------------------------------------------------------------------

	# -------------------------------------------------------------------
	#  Compute Log-likelihood of a dataset object given current model
	# -------------------------------------------------------------------
	# Computes the log-likelihood of data in any dataset object (either train or test) given the 
	# posterior predictive distributions over peaks and word-types for the model.
	# (cf Newman et al. (2009) "Distribution algorithms Topic Models" for standard LDA)
	# Note that this is not computing the joint log-likelihood of model parameters and data
	def computeLogLikelihood(self, dat, update_vectors = True):
		# --- Pre-compute all probabilities from count matrices that are needed for loglikelihood computations

		# Compute docprobs for y = ND x NT: p( y_i=t | d )
		doccounts = self.ny_d_t+self.alpha
		doccounts_sum = np.sum(doccounts, axis = 1 )
		docprobs_y = np.transpose( np.transpose(doccounts) / doccounts_sum )

		# Compute docprobs for z = ND x NT: p( z_i=t | y^(d) )
		doccounts = self.ny_d_t+self.gamma
		doccounts_sum = np.sum(doccounts, axis = 1 )
		docprobs_z = np.transpose( np.transpose(doccounts) / doccounts_sum )

		# Compute regionprobs = NR x NT: p( r | t )
		regioncounts = (self.ny_r_t) + self.delta
		regioncounts_sum = np.sum( regioncounts, axis = 0 )
		regionprobs = regioncounts / regioncounts_sum

		# Compute wordprobs = NW x NT: p( w | t )
		wordcounts = self.nz_w_t + self.beta
		wordcounts_sum = np.sum( wordcounts, axis = 0)
		wordprobs = wordcounts / wordcounts_sum

		# --- Get the matrix giving p(x_i|r,t) for all x:  
		#	NY x NT x NR matrix of probabilities of all peaks given all topic/subregion spatial distributions
		peak_probs = self.getPeakProbs(dat)

		# -----------------------------------------------------------------------------
		# --- Compute observed peaks (x) Loglikelihood: 
		# p(x|model, doc) = p(topic|doc) * p(subregion|topic) * p(x|subregion)
		# 			 	  = p_t_d * p_r_t * p_x_r
		x_loglikely = 0 # Initialize variable tracking total loglikelihood of all x tokens
		
		# Go over all observed peaks and add p(x|model) to running total
		for i in range(dat.npeaks):
			d = dat.peak_didx[i]-1 # convert didx from 1-idx to 0-idx
			p_x = 0 # Running total for p(x|d) across subregions: Compute p(x_i|d) for each subregion separately and then sum across the subregions
			for r in range(self.nr):
				p_t_d = docprobs_y[d] 			# p(t|d) - p(topic|doc)
				p_r_t = regionprobs[r] 			# p(r|t) - p(subregion|topic)
				p_r_d = p_t_d * p_r_t 			# p(r|d) - p(subregion|document) = p(topic|doc)*p(subregion|topic)
					
				p_x_r = peak_probs[i,:,r] 		# p(x|r) - p(x|subregion)
				p_x_rd = np.dot(p_r_d, p_x_r) 	# p(x|subregion,doc) = sum_topics ( p(subregion|doc) * p(x|subregion) )
				p_x += p_x_rd					# Add probability for current subregion to total probability for token across subregions
			x_loglikely += np.log(p_x)  		# Add probability for current token to running total for all x tokens
		
		# -----------------------------------------------------------------------------
		# --- Compute observed words (w) Loglikelihoods: 
		# p(w|model, doc) = p(topic|doc) * p(word|topic)
		# 			 	  = p_t_d * p_w_t
		w_loglikely = 0  # Initialize variable tracking total loglikelihood of all w tokens

		# Compute a matrix of posterior predictives over words: = ND x NW p(w|d) = sum_t ( p(t|d) * p(w|t) )
		p_w_d = np.dot(docprobs_z,np.transpose(wordprobs))

		# Go over all observed word tokens and add p(w|model) to running total
		for i in range(dat.nwords):
			w = dat.widx[i] - 1  	# convert widx  from 1-idx to 0-idx
			d = dat.wdidx[i] - 1 	# convert wdidx from 1-idx to 0-idx
			p_w = p_w_d[d,w]			# Probability of sampling current w token from d
			w_loglikely += np.log(p_w)	# Add log-probability of current token to running total for all w tokens

		# -----------------------------------------------------------------------------
		# --- Update model log-likelihood history vector (if update_vectors == True)
		if update_vectors:
			self.loglikely_iter.append(self.iter)	
			self.loglikely_x.append(x_loglikely)
			self.loglikely_w.append(w_loglikely)
			self.loglikely_tot.append(x_loglikely+w_loglikely)

		# ---------------------------------------------------------------------------------------------------------------
		# --- Return loglikely values (used when computing log-likelihood for a dataset-object containing hold-out data)
		return ( x_loglikely, w_loglikely, x_loglikely + w_loglikely)

	# ---------------------------------------------------------------------------------------------------------------
	# Compute a matrix giving p(x|r,t), using all x values in a dataset object, and each topic's spatial parameters
	# ---------------------------------------------------------------------------------------------------------------
	# 	NY x NT x NR matrix of probabilities, giving probability of sampling each peak (x) from all subregions
	def getPeakProbs(self, dat):
		peak_probs = np.zeros(shape = ( dat.npeaks, self.nt, self.nr), dtype = float)
		for t in range(self.nt):
			for r in range(self.nr):
				peak_probs[:,t,r] = multivariate_normal.pdf(dat.peak_vals , mean = self.regions_mu[t][r][0], cov = self.regions_sigma[t][r])
		return peak_probs

	# -------------------------------------------------------------------------------------------------------------------------------
	# Compute proportional multinomial probabilities of sampling observations x from all multinomial probability distribution rows p
	# -------------------------------------------------------------------------------------------------------------------------------
	# Note that this only returns values proportional to the true data-likelihood (sampler only requires proportionality)
	def mnpdf_proportional(self,x,p):
		# Inputs:
		#  x = a 1-by-T vector of observation
		#  p = A T-by-T matrix, where each row is one of the updated proposal distributions to y
		#  output: a 1-by-T vector giving the propostional probability of sampling x from each row of input y
		
		# First remove all columns for which we have no observations (dramatically speeds up computation)
		xPos = x>0
		x = x[xPos]
		p = p[:,xPos]
		
		# Now compute the probability sampling x from the row-probability vector (in log-space): product_i( p_i^(x_i) ), for all positive indices i 
		m, k = np.shape(p)
		x = np.dot(np.ones([m,1]),x.reshape([1, len(x)])) 
		xlogp = x*np.log(p)
		xlogp = np.sum(xlogp,axis=1) 		# Row-sums give total (proportional) of sampling vector x from rows of p 

		# Exponentiate to convert from log-space
		y = np.exp(xlogp - np.max(xlogp)) 	# Add a constant before exponentiating to avoid any underflow issues
		return y
	
	# -------------------------------------------------------------------------------------------------------------------------------
	# Compute proportional multinomial probabilities of current x vector given current y vector, for all proposed y_i values 
	# -------------------------------------------------------------------------------------------------------------------------------
	# Note that this only returns values proportional to the relative probabilities of all proposals for y_i 
	
	def compute_PropMultinomial_From_zy_Vectors(self,z,y):
		# Inputs:
		#  z = a 1-by-T vector of current z counts for document d
		#  y = a 1-by-T vector of current y counts (plus gamma) for document d
		#  output: a 1-by-T vector giving the proportional probability of z, given that topic t was incremented

		# Compute the proportional probabilities in log-space
		logp = z * np.log( (y+1) / y )
		p = np.exp(logp-np.max(logp));  # Add a constant before exponentiating to avoid any underflow issues
		return p

	# --------------------------------------------------------------------------------
	# <<<<< Export Methods >>>>> Print Topics, Model parameters, and Figures to file |
	# --------------------------------------------------------------------------------

	# -----------------------------------------------------------------------------
	# Run all export-methods: calls all print-methods to export parameters to files
	# -----------------------------------------------------------------------------
	def printAllModelParams(self,outputdir):
		# If output directory doesn't exist, make it
		if not os.path.isdir(outputdir):
			os.mkdir(outputdir)
		# Print topic-word distributions for top-K words in easy-to-read format
		outfilestr = outputdir + '/Topic_X_Word_Probs.csv' #^^^ adding '/'
		self.printTopicWordProbs(outfilestr, 20) 			
		# Print topic x word count matrix: m.nz_w_t
		outfilestr = outputdir + '/Topic_X_Word_CountMatrix.csv' #^^^ adding '/
		self.printTopicWordCounts(outfilestr) 
		# Print activation-assignments to topics and subregions: Peak_x, Peak_y, Peak_z, yidx, ridx
		outfilestr = outputdir + '/ActivationAssignments.csv' #^^^ adding '/
		self.printActivationAssignments(outfilestr)

	# -----------------------------------------------------------------------------
	# Print Peak->Topic and Peak->Subregion assignments for all x-tokens in dataset
	# -----------------------------------------------------------------------------
	def printActivationAssignments(self, outfilestr):
		# Open the file to print to
		fid = open(outfilestr,'w+')

		# Print the column-headers
		fid.write('Peak_X,Peak_Y,Peak_Z,Topic_Assignment,Subregion_Assignment')
		fid.write('\n')

		# For each peak-token, print out its coordinates and current topic/subregion assignment
		for i in range(self.ny):
			outstr = '%d,%d,%d,%d,%d' % (self.dat.peak_vals[i,0],
				self.dat.peak_vals[i,1],self.dat.peak_vals[i,2],
				self.yidx[i]+1, self.ridx[i]+1) # Note that we convert topic/subregion indices to 1-base idx
			fid.write(outstr)
			fid.write('\n')
		fid.close()

	# -----------------------------------------------------------------------------
	# Print Topic->Word counts for all topics and words
	# -----------------------------------------------------------------------------
	def printTopicWordCounts(self, outfilestr):
		# Open the file to print to
		fid = open(outfilestr,'w+')

		# Print the topic-headers
		fid.write('WordLabel,') # Header of wlabel column
		for t in range(self.nt):
			outstr = 'Topic_%02d,' % (t+1)
			fid.write(outstr)
		fid.write('\n')

		# For each row / wlabel: wlabel-string and its count under each topic (the \phi matrix before adding \beta and normalizing)
		for w in range(self.nw):
			# Print wlabel[w]
			outstr = '%s,' % self.dat.wlabels[w]
			fid.write(outstr)
			# Print counts under all topics
			for t in range(self.nt):
				outstr = '%d,' % self.nz_w_t[w,t]
				fid.write(outstr)
			# Newline for next wlabel row
			fid.write('\n')
		fid.close()

	# -----------------------------------------------------------------------------
	# Print Topic->Word probability distributions for top K words to File
	# -----------------------------------------------------------------------------
	def printTopicWordProbs(self, outfilestr, K = 15):
		# Open the file to print to
		fid = open(outfilestr, 'w+')

		# Compute topic->word probs and marginal topic-probs		 
		wprobs = self.nz_w_t + self.beta
		topic_probs = np.sum(wprobs,axis=0) / np.sum(wprobs) 	# Marginal topicprobs
		wprobs = wprobs /  np.sum(wprobs,axis=0) 				# Normalized word-probs

		# Get the sorted probabilities and indices of words under each topic
		rnk_vals = np.sort(wprobs, axis = 0)
		rnk_vals = rnk_vals[::-1]
		rnk_idx = np.argsort(wprobs,axis=0)
		rnk_idx = rnk_idx[::-1]

		# Print the topic-headers
		for t in range(self.nt):
			# Print each topic and it's marginal probability to columns
			outstr = 'Topic_%02d,%.4f,' % (t+1, topic_probs[t])
			fid.write(outstr)
		fid.write('\n')

		# Print the top K word-strings and word-probs for each topic
		for i in range(K):
			for t in range(self.nt):
				# Print the kth word in topic t and it's probability
				outstr = '%s,%.4f,' % (self.dat.wlabels[rnk_idx[i,t]], rnk_vals[i,t])
				fid.write(outstr)
			fid.write('\n')
		fid.close()

	# ----------------------------------------------------------------------------------------
	# Print Topic Figures: Spatial distributions and Linguistic distributions for top K words
	# ----------------------------------------------------------------------------------------
	# backgroundpeakfreq: Determines what proportion of peaks we show in the background of each figure
	def printTopicFigures(self, outputdir, backgroundpeakfreq = 10):
		# If output directory doesn't exist, make it
		if not os.path.isdir(outputdir):
			os.mkdir(outputdir)

		# Display parameters
		opts_axlims = [[-75, 75], [-110 ,90], [-60, 80]] # ^^ This would need to be changed for handling different data-types
		regioncolors = ['r','b','m','g','c','b']
		# Get a subset of values to use as background (to illustrate extent of all peaks)
		backgroundvals = self.dat.peak_vals[range(1,len(self.dat.peak_vals)-1, backgroundpeakfreq),:]
		backgroundvals = np.transpose(backgroundvals)
		# Loop over all topics and make a figure for each
		for t in range(self.nt):
			# Set up save file name (convert to base-1 indexing)
			outfilestr = '%s/Topic_%02d.png' % (outputdir, t+1)

			# Create figure
			fig = plt.figure(figsize=(10, 10), dpi=80)

			# <<<< Plot all points for topic from 3 different angles >>>>
			
			# --- REAR-VIEW: X-BY-Z
			ax1 = fig.add_subplot(221)
			ax1.axis('equal')
			# Plot background points in gray
			ax1.scatter(backgroundvals[0], backgroundvals[2],color='0.6', s=12, marker='o', alpha=.15)
			# Plot all subregion points in the subregion colors
			for r in range(self.nr):
				idx=(self.yidx==t) & (self.ridx==r)
				vals=self.dat.peak_vals[idx]
				valsplot = np.transpose(vals)
				ax1.scatter(valsplot[0], valsplot[2],c=regioncolors[r], s=12, lw=0, marker='^', alpha=.5)
			ax1.set_xlabel('X')
			ax1.set_ylabel('Z')
			ax1.set_xlim(opts_axlims[0])
			ax1.set_ylim(opts_axlims[2])
			
			# --- SIDE-VIEW: Y-BY-Z
			ax2 = fig.add_subplot(222)
			ax2.axis('equal')
			# Plot background points in gray
			ax2.scatter(backgroundvals[1], backgroundvals[2],color='0.6', s=12, marker='o', alpha=.15)
			# Plot all subregion points in the subregion colors
			for r in range(self.nr):
				idx=(self.yidx==t) & (self.ridx==r)
				vals=self.dat.peak_vals[idx]
				valsplot = np.transpose(vals)
				ax2.scatter(valsplot[1], valsplot[2],c=regioncolors[r], s=12, lw=0, marker='^', alpha=.5)
			ax2.set_xlabel('Y')
			ax2.set_ylabel('Z')
			ax2.set_xlim(opts_axlims[1])
			ax2.set_ylim(opts_axlims[2])

			# --- TOP-VIEW: X-BY-Y
			ax3 = fig.add_subplot(223)
			ax3.axis('equal')
			# --- Plot background points in gray
			ax3.scatter(backgroundvals[0], backgroundvals[1],color='0.6', s=12, marker='o', alpha=.15)
			# --- Plot all subregion points in the subregion colors
			for r in range(self.nr):
				idx=(self.yidx==t) & (self.ridx==r)
				vals=self.dat.peak_vals[idx]
				valsplot = np.transpose(vals)
				ax3.scatter(valsplot[0], valsplot[1],c=regioncolors[r], s=12, lw=0, marker='^', alpha=.5)
			ax3.set_xlabel('X')
			ax3.set_ylabel('Y')
			ax3.set_xlim(opts_axlims[0])
			ax3.set_ylim(opts_axlims[1])

			# <<<< Print words & Region-probs >>>>

			# ------- Get strings giving top K words and probs for the current topic ------
			k = 12
			wprobs = self.nz_w_t[:,t] + self.beta
			wprobs =  wprobs / np.sum(wprobs)
			# Get rankings of words
			wrnk = np.argsort(wprobs)
			wrnk = wrnk[::-1]

			# Create strings showing (1) top-K words (2) top-k probs for current topic
			outstr_labels = ''
			outstr_vals = ''
			for i in range(k):
				outstr_labels = outstr_labels + self.dat.wlabels[wrnk[i]] + '\n'
				outstr_vals = outstr_vals + "%5.3f" % wprobs[wrnk[i]] + '\n'

			# Fourth axis: Show top-k words and word-probs, then show region-probs
			ax4 = fig.add_subplot(224)
			ax4.set_xticklabels([])
			ax4.set_yticklabels([])
			ax4.set_yticks([])
			ax4.set_xticks([])
			ax4.set_title('Top k Words')
			plt.text(0.15, 0.98,outstr_labels, horizontalalignment='left',verticalalignment='top')
			plt.text(0.65, 0.98,outstr_vals, horizontalalignment='left', verticalalignment='top')

			# Now get subregion-probs for current topic
			rprobs = self.ny_r_t[:,t] + float(self.delta)
			rprobs = rprobs / sum(rprobs)

			# Print the region probs and means to axis
			outstr_region = 'Region-ID    p(r|t)     mu_1  mu_2  mu_3'
			plt.text(.03,.30, outstr_region, color = 'k', horizontalalignment='left', verticalalignment='top')
			for r in range(self.nr):
				outstr_region = 'Region %02d: %6.2f  |  %6.1f  %6.1f  %6.1f' % (r+1, rprobs[r],self.regions_mu[t][r][0][0],self.regions_mu[t][r][0][1],self.regions_mu[t][r][0][2])
				plt.text(.03,.22 -.06*r, outstr_region, color = regioncolors[r], horizontalalignment='left', verticalalignment='top')
			# Save figure to file and close it
			fig.savefig(outfilestr, dpi=fig.dpi)
			plt.close(fig)

	# -----------------------------------------------------------------------------------------
	# <<<<< Utility Methods for Displaying Model >>>> Display Model summary, Get Model-String |
	# -----------------------------------------------------------------------------------------

	# -----------------------------------------------------------------------------
	# Print model summary to console
	# -----------------------------------------------------------------------------
	def displayModelSummary(self):
		print "--- Model Summary ---" 
		print " Current State:" 
		print "\t Current iteration   = %d"   % self.iter
		print "\t Initialization Seed = %d"   % self.seed_init
		# ^^^ Update
		if len(self.loglikely_tot)>0:
			print "\t Current Log-Likely  = %d"   % self.loglikely_tot[-1]
		else:
			print "\t Current Log-Likely = ** Not Available: Model not yet initialized **"
		print " Model Hyper-Parameters:" 
		print "\t Symmetric = %s"   % self.symmetric
		print "\t nt    = %d" % self.nt
		print "\t nr    = %d" % self.nr
		print "\t alpha = %.3f" % self.alpha
		print "\t beta  = %.3f" % self.beta
		print "\t gamma = %.3f" % self.gamma
		print "\t delta = %.3f" % self.delta
		print "\t roi   = %.3f" % self.roi
		print "\t dobs  = %d" % self.dobs
		print " Model Training-Data Information:" 
		print "\t Dataset Label        = %s" % self.dat.datasetLabel
		print "\t # Word-Tokens (nz)   = %d" % self.nz
		print "\t # Peak-Tokens (ny)   = %d" % self.ny
		print "\t # Word-Types (nw)    = %d" % self.nw
		print "\t # Documents (nd)     = %d" % self.nd
		print "\t # Peak-Dimensions    = %d" % self.nxdims
		# print " DEBUG: Count matrices dimensionality:" 
		# print "\t ny_d_t   = %r" % (self.ny_d_t.shape,)
		# print "\t ny_r_t   = %r" % (self.ny_r_t.shape,)
		# print "\t nz_w_t   = %r" % (self.nz_w_t.shape,)
		# print "\t nz_sum_t = %r" % (self.nz_sum_t.shape,)
		# print "\t regions_mu    = %r" % (np.shape(self.regions_mu),)
		# print "\t regions_sigma = %r" % (np.shape(self.regions_sigma),)
		# print " DEBUG: Indicator vectors:" 
		# print "\t zidx   = %r" % self.zidx.shape
		# print "\t yidx   = %r" % self.yidx.shape
		# print "\t ridx   = %r" % self.ridx.shape
		# print " DEBUG: Sums (1):" 
		# print "\t sum(ny_d_t)   = %r" % np.sum(self.ny_d_t)
		# print "\t sum(ny_r_t)   = %r" % np.sum(self.ny_r_t)
		# print "\t sum(nz_w_t)   = %r" % np.sum(self.nz_w_t)
		# print "\t sum(nz_d_t)   = %r" % np.sum(self.nz_d_t)
		# print "\t sum(nz_sum_t) = %r" % np.sum(self.nz_sum_t)
		# print " DEBUG: Sums (2):" 
		# print "\t sum(ny_d_t, axis=0)   = %r" % np.sum(self.ny_d_t, axis=0)
		# print "\t sum(ny_r_t, axis=0)   = %r" % np.sum(self.ny_r_t, axis=0)

	# -------------------------------------------------------------------------
	# Get a model-string, unique to current datasetlabel + parameter settings |
	# -------------------------------------------------------------------------
	def getModelDisplayString(self):

		outstr = '%s_%dT_%dR_alpha%.3f_beta%.3f_gamma%.3f_delta%.3f_%ddobs_%.1froi_%dsymmetric_%d' % (self.dat.datasetLabel,
			self.nt, self.nr, self.alpha, self.beta, self.gamma, self.delta, self.dobs, self.roi, self.symmetric, self.seed_init)
		return outstr


if __name__=="__main__":
	print "Calling 'gclda_model.py' as a script"
else:
	print "Importing 'gclda_model.py' module v03"