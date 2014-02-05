#! /usr/bin/python

"""Training classifier

This script is used to train a classifier on the `training_set` and predict
it's classification accuracy on part of it.

Usage:
classifier.py /path/to/training/set.csv

Where the format of the training set csv file is defined by the class
formatted_io.TrainingSetIO

Output displays classification accuracy per number and displays the
misclassified images.
Like so:

                 precision    recall  f1-score   support

              0       0.91      0.93      0.92       150
              1       0.97      0.97      0.97       158
              2       0.89      0.89      0.89       144
              3       0.84      0.87      0.85       166
              4       0.85      0.88      0.87       138
              5       0.88      0.81      0.84       132
              6       0.93      0.92      0.92       155
              7       0.88      0.90      0.89       170
              8       0.80      0.81      0.80       135
              9       0.80      0.76      0.78       152

    avg / total       0.88      0.88      0.88      1500

    Displaying misclassified images.
    Close to finish script.
"""

import sys
from data_vis.image_display import ImageDisplay
from formatted_io import TrainingSetIO
from classifier.rbm_classifier import RBMClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split

training_parser = TrainingSetIO()
target_numbers, target_pixels = training_parser.parse(sys.argv[1])

training_set_pixels, testing_set_pixels, training_set_numbers, \
    testing_set_numbers = train_test_split(
        target_pixels, target_numbers, test_size=0.3
    )

classifier = RBMClassifier()
classifier.train(training_set_numbers, training_set_pixels)

test_set_predicted_numbers = classifier.predict(testing_set_pixels)

print metrics.classification_report(
    testing_set_numbers,
    test_set_predicted_numbers
)

image_plotter = ImageDisplay()
print("Displaying misclassified images...\nClose plot to finish script.")
plot = image_plotter.plot_incorrect_classifications(
    test_set_predicted_numbers,
    testing_set_numbers,
    testing_set_pixels
)
plot.show()
