# python_gclda

This is a Python implementation of the Generalized Correspondence-LDA model (gcLDA).

## Generalized Correspondence-LDA Model

This model is a generalization of the correspondence-LDA model (Blei & Jordan, 2003, "Modeling annotated data), which is used for modeling multiple data-types, where one data-type describes the other. This model was introduced in the following paper:

[PDF](mypage.iu.edu/~timrubin/Files/GC_LDA_Final.pdf)

where the model was applied for modeling fMRI publication data, consisting of the text and reported peak activation coordinates in the Neurosynth corpus of fMRI publications. In this context, the gcLDA model identifies a set of T topics, where each topic is associated with (1) a spatial probability distribution that captures the extent of a functional neural region, and (2) a probability distribution over linguistic features that describes the cognitive function of the region.

## Code Example

Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

## Motivation

A short description of the motivation behind the creation and maintenance of the project. This should explain **why** the project exists.

## Installation

Provide code examples and explanations of how to get the project.

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests

Describe and show how to run the tests with code examples.

## Contributors

Let people know how they can dive into the project, include important links to things like issue trackers, irc, twitter accounts if applicable.

## License

A short snippet describing the license (MIT, Apache, etc.)