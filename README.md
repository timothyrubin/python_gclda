# python_gclda

This is a Python implementation of the Generalized Correspondence-LDA model (gcLDA).

## Generalized Correspondence-LDA Model

This model is a generalization of the correspondence-LDA model (Blei & Jordan, 2003, "Modeling annotated data"), which is used for modeling multiple data-types, where one data-type describes the other. This model was introduced in the following paper:

[PDF](http://mypage.iu.edu/~timrubin/Files/GC_LDA_Final.pdf)

where the model was applied for modeling the Neurosynth corpus of fMRI publications. Each publication in this corpus consists of text and a set of reported peak activation brain coordinates. 

When applied to fMRI publication data, the gc-LDA model identifies a set of T topics, where each topic is associated with (1) a spatial probability distribution that captures the extent of a functional neural region, and (2) a probability distribution over linguistic features that describes the cognitive function of the region.

The gcLDA model can additionally be directly applied to other types of data. For example, Blei & Jordan presented correspondence-LDA for modeling annotated images, where pre-segmented images were represented by vectors of real-valued image features.

## Summary of code

This repository consists of two python classes (contained within the subdirectory 'Python_gclda/gc_lda'), several scripts that illustrate how to uses these classes to train and export a gcLDA model (contained within the subdirectory 'Python_gclda/'), and properly formatted versions of the Neurosynth dataset that the model can be applied to (contained within the subdirectory 'Data/Datasets').



## Citing the code and data

Please cite the following paper if you wish to reference this code:

Rubin, Timothy N., Koyejo, Oluwasanmi, Jones, Michael N., Yarkoni, Tal, (Submitted). Generalized Correspondence-LDA Models (GC-LDA) for Identifying Functional Regions in the Brain.

To reference any of the datasets contained in this repository, please cite the following:

Yarkoni, Tal, Russell A. Poldrack, Thomas E. Nichols, David C. Van Essen, and Tor D. Wager. "Large-scale automated synthesis of human functional neuroimaging data." Nature methods 8, no. 8 (2011): 665-670.

For additional details about the dataset please visit [neurosynth.org](http://http://neurosynth.org/)
