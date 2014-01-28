#! /usr/bin/python

import sys
from formatted_io import TrainingSetIO
from memory_profiler import profile
from classifier.rbm_classifier import RBMClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split


@profile
def foo():
    training_parser = TrainingSetIO()

    target_numbers, target_pixels = training_parser.parse(sys.argv[1])

    training_set_pixels, testing_set_pixels, training_set_numbers, testing_set_numbers = train_test_split(
        target_pixels, target_numbers, test_size = 0.3
    )

    classifier = RBMClassifier()

    classifier.train(training_set_numbers, training_set_pixels)

    print metrics.classification_report(
        testing_set_numbers,
        classifier.predict(testing_set_pixels)
    )

    classifier.plot_rbm_features()

if __name__ == "__main__":
    foo()
