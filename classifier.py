#! /usr/bin/python

import sys
from formatted_io.input_parser import InputParser
from memory_profiler import profile
from numpy import zeros, uint8, float16
from classifier.rbm_classifier import RBMClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split


@profile
def foo():
    input_parser = InputParser()

    target_pixels = zeros((42000, 784), float16)
    target_numbers = zeros((42000, ), uint8)

    sample_count = 0
    for sample_data in input_parser.parse_train(sys.argv[1]):
        pixels = uint8(sample_data[1])

        target_numbers[sample_count] = sample_data[0]
        target_pixels[sample_count] = pixels

        sample_count += 1

    # 0-1 scaling
    target_pixels /= 255

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
