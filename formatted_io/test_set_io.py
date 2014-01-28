from numpy import uint8
from formatted_io import InputParser

"""Parses kaggle input test set data for processing

For training data input is expected in the form:

pixel0, pixel1, ... pixelN
22,33,...250
53,0,...125
...

Where the label column is absent for testing, the format is otherwise identical

Pixels can range from 0 -> 255
"""
class TestSetIO(InputParser):
    num_columns = 784

    def parse(self, file_path):
        """Parses the test set for pixel values

        Generates arrays of pixel values
        """

        pixels = []

        for sample_data in super(TestSetIO, self).parse(file_path):
            pixels.append(uint8(sample_data))

        return super(TestSetIO, self)._process_raw_pixel_values(pixels)

    def _check_headings(self, csv_reader):
        headings = csv_reader.next()
        super(TestSetIO, self)._check_heading_length(headings)


