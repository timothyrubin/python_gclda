# python_gclda

This is a Python implementation of the Generalized Correspondence-LDA model (gcLDA).

## Generalized Correspondence-LDA Model (GC-LDA)

The gcLDA model is a generalization of the correspondence-LDA model (Blei & Jordan, 2003, "Modeling annotated data"), which is an unsupervised learning model used for modeling multiple data-types, where one data-type describes the other. The gcLDA model was introduced in the following paper:

[Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain](http://mypage.iu.edu/~timrubin/Files/GC_LDA_Final.pdf)

where the model was applied for modeling the Neurosynth corpus of fMRI publications. Each publication in this corpus consists of a set of word tokens and a set of reported peak activation coordinates (x, y and z spatial coordinates corresponding to brain locations). 

When applied to fMRI publication data, the gcLDA model identifies a set of T topics, where each topic captures a 'functional region' of the brain. More formally: each topic is associated with (1) a spatial probability distribution that captures the extent of a functional neural region, and (2) a probability distribution over linguistic features that captures the cognitive function of the region.

The gcLDA model can additionally be directly applied to other types of data. For example, Blei & Jordan presented correspondence-LDA for modeling annotated images, where pre-segmented images were represented by vectors of real-valued image features. The code provided here should be directly applicable to these types of data, provided that they are appropriately formatted.

## Installation

Dependencies for this package are: scipy, numpy and matplotlib. If you don't have these installed, these easiest way to do so might be to use [Anaconda](https://www.continuum.io/downloads). Alternatively, [this page](http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/) gives a useful tutorial on installing them.

Additionally, some of the example scripts rely on gzip and cPickle (for saving compressed model instances to disk).

This code can be installed as a python package using:

	> python setup.py install

The classes needed to run a gclda model can then be imported into python using:

	> from python_gclda_package.gclda_dataset import gclda_dataset

	> from python_gclda_package.gclda_model   import gclda_model


## Summary of python_gclda package

The repository consists of: 

- two python classes (contained within the subdirectory 'python_gclda/python_gclda_package')
- several scripts and a tutorial that illustrate how to uses these classes to train and export a gcLDA model (contained within the subdirectory 'python_gclda/examples')
- formatted versions of the Neurosynth dataset that can be used to train a gclda model (contained within the subdirectory 'python_gclda/datasets/neurosynth')
- some examples of results from trained gclda models under different parameter settings (contained within subdirectories of 'python_gclda/example_results').

## Dataset formatting

The gclda_dataset class requires four .txt files containing all dataset features that the gcLDA model needs to operate. Please see the example datasets in the 'Data/Datasets' subdirectory, for examples of properly formatted data. For additional details about these files, please see the 'README.txt' file in the 'python_gclda/' subdirectory.

## Tutorial usage examples

For a simple tutorial showing most uses of the gclda code, see the following file:
	'python_gclda/examples/tutorial_gclda.py'
This script illustrates how to (1) build a dataset object (using a small subset of the neurosynth dataset), (2) train a gclda model on the dataset object, and (3) export figures illustrating the trained model to files for viewing.

## Code usage examples

For additional examples of how to use the code, please see the following scripts in the 'python_gclda/examples' subdirectory:

- 'script_run_gclda.py': Illustrates how to build a dataset object from a full formatted version of the full neurosynth dataset, and then train a gcLDA model (using the dataset object and several hyperparameter settings that get passed to the model constructor)
- 'export_model_figs.py': Illustrates how to export model data, and png files illustrating each topic, using a trained gcLDA model.

Additional details about the gcLDA code, gcLDA hyper-parameter settings, and about these scripts is provided in the 'README.txt' in the 'Python_gclda/' subdirectory. Note that all three models presented in the source paper ('no subregions', 'unconstrained subregions' and 'constrained subregions') can be trained by modifying the model hyper-parameters

## Example results for trained models

Results for some example trained models (including .png files illustrating all topics for the models) are included in the 'python_gclda/example_results' subdirectories.

## Citing the code and data

Please cite the following paper if you wish to reference this code:

Rubin, Timothy N., Koyejo, Oluwasanmi, Jones, Michael N., Yarkoni, Tal, (Submitted). Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain.

To reference any of the datasets contained in this repository or Neurosynth:

Yarkoni, Tal, Russell A. Poldrack, Thomas E. Nichols, David C. Van Essen, and Tor D. Wager. "Large-scale automated synthesis of human functional neuroimaging data." Nature methods 8, no. 8 (2011): 665-670.

For additional details about Neurosynth please visit [neurosynth.org](http://neurosynth.org/)
