from csv import reader
from pandas import Series, SparseDataFrame
from numpy import uint8

"""Parses kaggle input training and test set data for further processing

For training data input is expected in the form:

label, pixel0, pixel1, ... pixelN
1,22,33,...250
9,53,0,...125
...

Where the label column is absent for testing, the format is otherwise identical

Pixels can range from 0 -> 255
Labels can range from 0 -> 9
"""
class InputParser(object):
	num_pixels = 784

	def parse_test(self, file_path):
		"""Parses the test set for pixel values

		Generates arrays of pixel values
		"""

		pixels = []

		with open(file_path, "rU") as sample:
			csv_reader = reader(sample)

			# Note: consumes the heading row
			self._check_test_set_headings(csv_reader)

			for sample_data in csv_reader:
				pixels.append(uint8(sample_data))

			pixels = SparseDataFrame(pixels, default_fill_value=0)
			pixels /= 255

			return pixels

	def parse_train(self, file_path):
		"""Parses the training set for labels (image numbers) and pixel values

		Generates tuples of integer labels and arrays of pixel values
		"""

		pixels = []
		numbers = []

		with open(file_path, "rU") as sample:
			csv_reader = reader(sample)

			# Note: consumes the heading row
			self._check_training_set_headings(csv_reader)

			for sample_data in csv_reader:
				numbers.append(uint8(sample_data[0]))
				pixels.append(uint8(sample_data[1:]))

			pixels = SparseDataFrame(pixels, default_fill_value=0)
			# Normalising to 0-1 range
			pixels /= 255

			# @TODO: Check performance of series. Here for consistency
			return (Series(numbers), pixels)

	def _check_training_set_headings(self, csv_reader):
		headings = csv_reader.next()

		if (headings[0] != "label"):
			raise TypeError(
				"Training set must have a label field as it's first value"
			)
		elif (len(headings) != self.num_pixels + 1):
			raise TypeError(
				"Training set expects have %d columns" % (self.num_pixels + 1)
			)

	def _check_test_set_headings(self, csv_reader):
		headings = csv_reader.next()
		len_headings = len(headings)
		if (len_headings != self.num_pixels):
			raise TypeError(
				"Training set expects to have %d columns. %d encountered" %
				(self.num_pixels, len_headings)
			)


