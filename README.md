# python_gclda

This is a Python implementation of the Generalized Correspondence-LDA model (gcLDA).

## Generalized Correspondence-LDA Model

This model is a generalization of the correspondence-LDA model (Blei & Jordan, 2003, "Modeling annotated data"), which is used for modeling multiple data-types, where one data-type describes the other. This model was introduced in the following paper:

[PDF](http://mypage.iu.edu/~timrubin/Files/GC_LDA_Final.pdf)

where the model was applied for modeling the Neurosynth corpus of fMRI publications. Each publication in this corpus consists of text and a set of reported peak activation brain coordinates. 

When applied to fMRI publication data, the gc-LDA model identifies a set of T topics, where each topic is associated with (1) a spatial probability distribution that captures the extent of a functional neural region, and (2) a probability distribution over linguistic features that describes the cognitive function of the region.

The gcLDA model can additionally be directly applied to other types of data. For example, Blei & Jordan presented correspondence-LDA for modeling annotated images, where pre-segmented images were represented by vectors of real-valued image features. The code provided here should be directly applicable to these types of data, provided that they are appropriately formatted.

## Summary of code

This repository consists of two python classes (contained within the subdirectory 'Python_gclda/gc_lda'), several scripts that illustrate how to uses these classes to train and export a gcLDA model (contained within the subdirectory 'Python_gclda/'), and formatted versions of the Neurosynth dataset that the model can be applied to (contained within the subdirectory 'Data/Datasets').

## Dataset formatting

The gclda_dataset class requires four .txt files containing all dataset features that the gcLDA model needs to operate. Please see the example datasets in the 'Data/Datasets' subdirectory, for examples of properly formatted data. For additional details about these files, please see the 'README.txt' file in the 'Python_gclda/' subdirectory.


## Code usage examples

For specific examples of how to use the code, please see the following scripts in the 'Python_gclda' subdirectory:

- 'run_gclda.py': Illustrates how to build a dataset object from formatted fMRI publication data, and then train a gcLDA model (using the dataset object and several hyperparameter settings that get passed to the model constructor)
- 'export_model_figs.py': Illustrates how to export model data, and png files illustrating each topic, using a trained gcLDA model.

Additional details about the gcLDA code, gcLDA hyper-parameter settings, and about these scripts is provided in the 'README.txt' in the 'Python_gclda/' subdirectory. Note that all three models presented in the source paper ('no subregions', 'unconstrained subregions' and 'constrained subregions') can be trained by modifying the model hyper-parameters

## Example results for trained models

Results for some example trained models (including .png files illustrating all topics for the models) are included in the 'Python_gclda/Python_gclda/gclda_results' subdirectories.

## Citing the code and data

Please cite the following paper if you wish to reference this code:

Rubin, Timothy N., Koyejo, Oluwasanmi, Jones, Michael N., Yarkoni, Tal, (Submitted). Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain.

To reference any of the datasets contained in this repository, or the Neurosynth dataset in general, please cite the following:

Yarkoni, Tal, Russell A. Poldrack, Thomas E. Nichols, David C. Van Essen, and Tor D. Wager. "Large-scale automated synthesis of human functional neuroimaging data." Nature methods 8, no. 8 (2011): 665-670.

For additional details about the dataset please visit [neurosynth.org](http://neurosynth.org/)
