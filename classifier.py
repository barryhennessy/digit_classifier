#! /usr/bin/python

import sys
from formatted_io.input_parser import InputParser
from memory_profiler import profile
from numpy import zeros, int8
from classifier.rbm_classifier import RBMClassifier
from sklearn.cross_validation import train_test_split
from sklearn import datasets

@profile
def foo():
	input_parser = InputParser()

	target_pixels = zeros((28000, 783), bool)
	target_numbers = zeros((28000, 1), int8)

	sample_count = 0
	for sample_data in input_parser.parse_train(sys.argv[1]):
		target_numbers[sample_count] = sample_data[0]
		target_pixels[sample_count] = bool(sample_data[1])
		++sample_count

	training_set_pixels, testing_set_pixels, training_set_numbers, testing_set_numbers = train_test_split(
		target_pixels, target_numbers, test_size = 0.3
	)
		
	classifier = RBMClassifier()
	classifier.train(training_set_numbers, training_set_pixels)

	print classifier.predict(testing_set_numbers)

def dataset_inspection():
	digits = datasets.load_digits()
	print(digits.data)
	print(digits.data.shape)
	print(digits.target)
	print(digits.target.shape)

foo()

dataset_inspection()