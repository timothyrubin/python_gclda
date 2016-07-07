--- Overview ----
The two classes that this code uses are:
	gclda_dataset: this class stores all neurosynth data used by the gclda model, and has methods for importing this data from .txt files into the python object
	gclda_model: this class gets passed a dataset object and all gclda hyperparameter settings, and creates a gclda_model object. The model object has methods for performing parameter updates (gibbs sampling of topic-assignment, estimates of spatial parameters), as well as exporting model parameters to files. 

--- Dataset class info --- 
	The first thing that the "Run_gcLDA_v02.py" script does is build a dataset object, using the "gclda_dataset" class. This dataset object is then passed to the gclda_model object.
	- The gclda_dataset class constructor needs to be passed 2 arguments: a 'datasetlabel' string that gives the name of the directory holding the formatted neurosynth data, and a 'datadirectory' string that points from the working directory to the root directory of the 'datasetlabel'. These are separated because the 'datasetlabel' will be used in the filename when saving gclda objects to disk, so that each save-file is associated with the dataset version
	- The gclda_dataset class requires 4 txt files with formatted neurosynth data to be in the 'datasetlabel' directory, in order to build a dataset object. There is an example of a formatted dataset (using the most recent version of the neurosynth dataset that I have from you), where you can see the formatting that the code expects. The four input files expected are as follows:
		- peakindices.txt: Contains document indices (1-to-D, where the doc-index i corresponds to the ith pubmed-id in the 'pmids.txt' file), and x/y/z values of peak-activation coordinates, for each peak activation token in the dataset
		- wordindices.txt: Contains document indices (1-to-D), and word-indices (1-to-W, where the jth index corresponds to the jth word-type in the 'wordlabel.txt' file) for all word tokens in the dataset
		- pmids.txt: a list of pubmed ids for all documents
		- wordlabels.txt: a list of word-strings for all word-types.

--- Scripts ---
The following two scripts are ready to run out-of-the box. 
	- "run_gclda.py: used to train a gc-lda model. 
	- "export_model_figs.py": used to export a trained model to .csv files (in the same format I've sent to you previously), and print some ugly figures.
Details about running scripts/parameters are below.

There is one additional script that illustrates how the code can be used to evaluate models in terms of the log-likelihood of hold-out data. Probably not important but let me know if you'd like clarification.

--- Run_gcLDA.py ---
	- This script is used to run the GC-LDA model. If starting a new model, it builds a dataset object (using the class in gc_LDA dataset), and then a model object (using the class in gcLDA_model). The model object constructor gets passed a dataset object and the gc-lda parameters. The script will then run the model update methods for any number of iterations.
	- As model iterates over parameter update methods, the code saves a compressed model and the topic->word distributions every 'save_freq' iterations. These are saved in directory 'gclda_results', in a subfolder whose name corresponds to a description of the model/dataset. 
	- The script can be used to resume updates of a model from any of the save-points, so the model saves are useful in case of a crash or if you need to halt the program. (it is safe to delete all save-points but the most recent). 
	-- Parameters --
	- 'Current_iter' parameter. If running a new model, set current_iter = 0, which tells the script to create a new dataset object and model object. If you want to continue sampling from a saved model, just set 'current_iter' to the iteration of the last saved model instance (and make sure all other parameter settings match the saved model settings). This is useful if you ever want to run sampling in chunks (or if you ever want to halt the process while the sampler is running)
	- Dataset Parameters: the 'datasetLabel' parameter is used to track the name/version of the dataset, and should correspond to the directory containing the 4 txt files described above. 'dataDirectory' is a string pointing to the directory containing the 'datasetLabel' directory
	- Model Parameters: 
		- The gclda model parameter settings are configured at the top of this script, and each parameter is commented explaining its role in the model. The script is currently configured to run a symmetric gc-lda model with 2 subregions and 200 topics, but can be used to run any model variant. E.g., to run an unconstrained subregions model with 100 topics, change parameters: "nt=100" and "symmetric=false". To run a no-subregions model, set "nr=1" and "symmetric=false" (symmetric only works for a model with 2 subregions)
		- Additional note on model parameters: There are a two hyperparameters that were used in the model but not discussed in the nips paper ('roi' and 'dobs'). These are used by the 'updateRegions' method, and are used to help regularize the estimates of covariance matrices for spatial distributions, which helps ensure stability of these estimates in cases where very few peaks get assigned to a subregion. Specifically:
			- 'roi' gives a "default width" for each regions covariance matrix. The covariances matrices will be nudged towards being a diagonal matrix with the value 'ROI' along the diagonals. Set this to what would be considered an'ideal' average region size
			- 'dobs' acts as a count of pseudo-observations for each subregion's spatial distribution, where these pseudo-observations are assumed to have the default covariance matrix. If 'dobs' is much greater than the number of actual peaks assigned to a subregion, the subregion's covariance matrix will end up being nearly equal to the 'default' matrix, defined by the 'roi' parameter
	- Sampling Parameters (unrelated to model specification but used for inference):
		- 'total_iterations': The total number of iterations to (stops sampling when model.iter == total_iterations) A broad recommendation for running this code is to run for at least 1000 total iterations (although the general focus of many topics becomes apparent much earlier, often after just 50 or so iterations).
		- 'save_freq': Determines how often a compressed gclda model and topic->word distributions get saved to disk.
		- 'loglikely_freq': Determines how often model loglikelihood on training data is computed (useful for tracking model, but slows things down if computed every iteration)
		- 'sampler_verbosity': Controls how much sampling information is printed to console (valid values are 0 for minimum and 2 for maximum)

--- gcLDA_exportModel_printFigures.py ---
	- This script is used to export model parameters, and print figures illustrating topics from a saved model. It loads the model from a savefile, and then exports the relevant model parameters to csv files. The exported csv files are formatted exactly the same as the ones that I've been passing to you previously, with files 'ActivationAssignments.csv', 'Topic_X_Word_Probs.csv', and 'Topic_X_Word_CountMatrix.csv'). It also prints figures illustrating each topic.
	- The only changes this script needs are to make sure of the following:
		- The datasetlabel and gclda parameter settings match those for the model you wish to export
		- The 'current_iter' is set to the final *saved* iteration of the model

--- predict_holdout_data.py ---
	- This script shows how to evaluate the loglikelihood of a hold-out dataset using a trained gclda model. (it loads a model trained on a training dataset, from which holdout data was removed, and evaluates predictions on the holdout data)


--- Datasets ---
	- There are three preprocessed datasets in the '/Data/' directory. 
		- '2015Filtered2': This is the most recent version of the full neurosynth dataset that I have from you
		- '2015Filtered2_TrnTst1P1': This is the 'training' dataset I used for experiments with a train/test set for the NIPS paper (where 1/5th of peak and word tokens are removed from each document)
		- '2015Filtered2_TrnTst1P2': This is the 'testing' dataset I used for experiments with a train/test set for the NIPS paper (containing all of the data removed from the training set).
		- These second two datasets are there just to illustrate how the model can be used for predicting hold-out data.
