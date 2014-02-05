#! /usr/bin/python

"""Full classifier

This script trains the classifier with the full training set and predicts
values on the test set. The predicted values are written to the path given as
the third argument

Usage:
test_set_classifier.py /path/to/training.csv /path/to/test.csv /output/path.csv

Formats of the training set, test set and output are as defined in
TrainingSetIO and TestSetIO
"""

import sys
from formatted_io import TrainingSetIO, TestSetIO
from classifier.random_forest_classifier import RandomForestClassifier

training_parser = TrainingSetIO()

target_numbers, target_pixels = training_parser.parse(sys.argv[1])

classifier = RandomForestClassifier()
classifier.train(target_numbers, target_pixels)

test_set_parser = TestSetIO()
test_set = test_set_parser.parse(sys.argv[2])

prediction = classifier.predict(test_set)

# @TODO: Why on earth doesn't this write to stdout??
test_set_parser.write(prediction, sys.argv[3])
