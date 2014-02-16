# Digit classification
This project is built to test various methods of classifying digits from the MNIST digit classification challenge. It is set up so that various classifiers and/or pipelines of various classifiers can be run through the test and their performance compared.

Some classifiers here have been submitted to the kaggle digit recogniser competition, scoring a respectable 96% classification accuracy. 


### Usage & key points
`classifier.py` is used to predict the classification accuracy, time and memory performance of a classifier and to run
classification against a test set. See `classifier.py -h` for full details.

Classifiers extend `classifier.classifier` and implement a single method to set up and return the classifier/pipeline.

Classifiers can also be used to do a grid search of their parameter space. This is useful to find the right parameters for the particular classification task at hand. Although it can take *some time* to complete a grid search of many combinations.

Basic visualisation of the digits is available in the `data_vis` sub package.

Please note: with large datasets and/or excessive parameters these classifiers can take a *long* time to execute and can hog much of your machine while doing so. Be careful if you don't want to leave your computer alone for a while.
