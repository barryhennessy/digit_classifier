#! /usr/bin/env python

"""Digit classifier

This script is used to train a classifier on the `training_set` and optionally
predict it's classification accuracy on part of it. If a `test_set` is provided
the entire training set is used to build the classifier and the digits are
predicted for the entire test_set.

Usage:

To test classifiers against test sets of known numbers
    classifier.py /path/to/training/set.csv

Output displays classification accuracy per number and displays the
misclassified images.

To test classifiers against test sets of unknown numbers
    classifier.py /path/to/training/set.csv --test-set-path="/path/to/test/set.csv"

Outputs predicted numbers, with an index to stdout.
"""

from argparse import ArgumentParser
from sys import stdout

from classifier.random_forest_classifier import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split

from data_vis.image_display import ImageDisplay
from formatted_io import TrainingSetIO, TestSetIO

if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        metavar='training/set/path',
        dest="training_set",
        help="the path to the training set"
    )

    arg_parser.add_argument(
        "--test-set-path",
        dest="test_set",
        help="the path to the test set"
    )

    arg_parser.add_argument(
        "--show-misclassified",
        dest="show_misclassified",
        help="display misclassified numbers",
        action="store_true"
    )

    args = arg_parser.parse_args()

    training_parser = TrainingSetIO()
    training_set_numbers, training_set_pixels = training_parser.parse(
        args.training_set
    )

    if args.test_set:
        training_set_numbers = training_set_numbers

        test_set_parser = TestSetIO()
        test_set_pixels = test_set_parser.parse(args.test_set)
    else:
        # Without a test set we need to split one out from the training set
        training_set_pixels, test_set_pixels, training_set_numbers, \
            test_set_numbers = train_test_split(
                training_set_pixels, training_set_numbers, test_size=0.3
            )

    classifier = RandomForestClassifier()
    classifier.train(training_set_numbers, training_set_pixels)

    test_set_predicted_numbers = classifier.predict(test_set_pixels)

    if args.test_set:
        test_set_parser.write(test_set_predicted_numbers, stdout)
    else:
        print metrics.classification_report(
            test_set_numbers,
            test_set_predicted_numbers
        )

        if args.show_misclassified:
            image_plotter = ImageDisplay()
            print(
                "Displaying misclassified images...\nClose plot to finish "
                "script."
            )
            plot = image_plotter.plot_incorrect_classifications(
                test_set_predicted_numbers,
                test_set_numbers,
                test_set_pixels
            )
            plot.show()
