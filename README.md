# python_gclda

This is a Python implementation of the Generalized Correspondence-LDA model (gcLDA).

## Generalized Correspondence-LDA Model (GC-LDA)

The gcLDA model is a generalization of the correspondence-LDA model (Blei & Jordan, 2003, "Modeling annotated data"), which is an unsupervised learning model used for modeling multiple data-types, where one data-type describes the other. The gcLDA model was introduced in the following paper:

[Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain](https://timothyrubin.github.io/Files/GCLDA_NIPS_2016_Final_Plus_Supplement.pdf)

where the model was applied for modeling the [Neurosynth](http://neurosynth.org/) corpus of fMRI publications. Each publication in this corpus consists of a set of word tokens and a set of reported peak activation coordinates (x, y and z spatial coordinates corresponding to brain locations). 

When applied to fMRI publication data, the gcLDA model identifies a set of T topics, where each topic captures a 'functional region' of the brain. More formally: each topic is associated with (1) a spatial probability distribution that captures the extent of a functional neural region, and (2) a probability distribution over linguistic features that captures the cognitive function of the region.

The gcLDA model can additionally be directly applied to other types of data. For example, Blei & Jordan presented correspondence-LDA for modeling annotated images, where pre-segmented images were represented by vectors of real-valued image features. The code provided here should be directly applicable to these types of data, provided that they are appropriately formatted. Note however that this package has only been tested on the Neurosynth dataset; some modifications may be needed for use with other datasets. 

## Installation

Dependencies for this package are: scipy, numpy and matplotlib. If you don't have these installed, the easiest way to do so may be to use [Anaconda](https://www.continuum.io/downloads). Alternatively, [this page](http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/) provides a tutorial on installing them (note that the line "brew install gfortran" now must be replaced by "brew install gcc").

Additionally, some of the example scripts rely on gzip and cPickle (for saving compressed model instances to disk).

This code can be installed as a python package using:

	> python setup.py install

The classes needed to run a gclda model can then be imported into python using:

	> from python_gclda_package.gclda_dataset import gclda_dataset

	> from python_gclda_package.gclda_model   import gclda_model


## Summary of python_gclda package

The repository consists of: 

- two python classes (contained within the subdirectory 'python_gclda_package')
- several scripts and a tutorial that illustrate how to uses these classes to train and export a gcLDA model (contained within the subdirectory 'examples')
- formatted versions of the Neurosynth dataset that can be used to train a gclda model (contained within the subdirectory 'datasets/neurosynth')
- some examples of results from trained gclda models under different parameter settings (contained within subdirectories of 'example_results').

## Dataset formatting

The gclda_dataset class requires four .txt files containing all dataset features that the gcLDA model needs to operate. Please see the example datasets in the 'Data/Datasets' subdirectory, for examples of properly formatted data. For additional details about these files, please see the 'README.txt' file in the 'documentation' subdirectory.

## Tutorial usage examples

For a simple tutorial illustrating usage of the gclda package, see the following file:
- 'examples/tutorial_gclda.py'

This tutorial demonstrates how to (1) build a dataset object (using a small subset of the neurosynth dataset), (2) train a gclda model on the dataset object, and (3) export figures illustrating the trained model to files for viewing.

There is also a version of this same tutorial in the following Jupyter notebook:
- 'examples/tutorial_gclda_notebook.ipynb'

## Code usage examples

For additional examples of how to use the code, please see the following scripts in the 'examples' subdirectory:

- 'script_run_gclda.py': Illustrates how to build a dataset object from a version of the neurosynth dataset, and then train a gcLDA model (using the dataset object and several hyperparameter settings that get passed to the model constructor).
- 'script_export_gclda_figs.py': Illustrates how to export model data and png files illustrating each topic from a trained gcLDA model object.
- 'script_predict_holdout_data.py': Illustrates how to compute the log-likelihood for a hold-out dataset.

Note that these scripts operate on the following version of the neurosynth dataset: "2015Filtered2_TrnTst1P1", which is a training dataset from which a subset document data has been removed for testing (the test-data is in the dataset: "2015Filtered2_TrnTst1P2"). The complete neurosynth dataset, without any test-data removed, is the version labeled "2015Filtered2".

Additional details about the gcLDA code, gcLDA hyper-parameter settings, and about these scripts are provided in the 'README.txt' in the 'documentation' subdirectory, as well as in the comments of the 'script_run_gclda.py' file. Note that all three models presented in the source paper ('no subregions', 'unconstrained subregions' and 'constrained subregions') can be trained by modifying the model hyper-parameters appropriately.

## Example results for trained models

Results for some example trained models (including .png files illustrating all topics for the models) are included in the 'example_results' subdirectories.

## Using alternative spatial distributions

As described in our paper, the gcLDA model allows one to associate topics with any valid probability distribution for modeling the observed 'x' data. The package currently has the ability to train gcLDA models using Gaussian mixture models with any number of components, as well as Gaussian mixture models with spatial constraints. If you wish to modify the code to train a model using an alternative distribution, you will need to modify the following methods in gclda_model.py: (1) update_regions (2) getPeakProbs, as well as the lines of the (3) \_\_init__ method which allocate memory for storing the distributional parameters. 

## Citing the code and data

Please cite the following paper if you wish to reference this code:

- Timothy N Rubin, Oluwasanmi Koyejo, Michael N Jones, Tal Yarkoni (Submitted). Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain.

Additionally, the following paper demonstrates a variety of cool applications for gcLDA models trained on Neurosynth (such as "brain decoding"):

- Timothy N Rubin, Oluwasanmi Koyejo, Krzysztof J Gorgolewski, Michael N Jones, Russell A Poldrack, Tal Yarkoni (Submitted). [Decoding brain activity using a large-scale probabilistic functional-anatomical atlas of human cognition](http://biorxiv.org/content/early/2016/06/18/059618)

To reference any of the datasets contained in this repository, or Neurosynth itself:

- Tal Yarkoni, Russell A. Poldrack, Thomas E. Nichols, David C. Van Essen, and Tor D. Wager. "Large-scale automated synthesis of human functional neuroimaging data." Nature methods 8, no. 8 (2011): 665-670.

Additionally, the complete Neurosynth datasets can be accessed at http://github.com/neurosynth/neurosynth-data (note however that those datasets need to be reformatted in order to make them work with the python_gclda package)

For additional details about Neurosynth please visit [neurosynth.org](http://neurosynth.org/)
