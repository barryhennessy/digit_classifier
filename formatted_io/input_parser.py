from csv import reader

"""Parses kaggle input training and test set data for further processing

Input is expected in the form:

label, pixel0, pixel1, ... pixelN
1,22,33,...250
9,53,0,...125
...

Where the label column is optional
"""
class InputParser(object):
	num_pixels = 784

	def parse_test(self, file_path):
		"""Parses the test set for and pixel values

		Generates arrays of pixel values
		"""

		with open(file_path, "rU") as sample:
			csv_reader = reader(sample)

			# Note: consumes the heading row
			self._check_test_set_headings(csv_reader)

			for sample_data in csv_reader:
				yield sample_data

	def parse_train(self, file_path):
		"""Parses the training set for labels (image numbers) and pixel values

		Generates tuples of integer labels and arrays of pixel values

		@TODO: Use native numpy sparse arrays?
		"""

		with open(file_path, "rU") as sample:
			csv_reader = reader(sample)

			# Note: consumes the heading row
			self._check_training_set_headings(csv_reader)

			for sample_data in csv_reader:
				yield self._format_training_set(sample_data)			

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
	def _format_training_set(self, csv_data):
		return (csv_data[0],  csv_data[1:])
		return [csv_data[0], csv_data[1:]]

	def _check_test_set_headings(self, csv_reader):
		headings = csv_reader.next()

		if (len(headings) != self.num_pixels):
			raise TypeError(
				"Training set expects to have %d columns" % (self.num_pixels)
			)


