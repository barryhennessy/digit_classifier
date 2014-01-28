from numpy import uint8, zeros
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

        num_rows = self._parse_file_data_size(file_path)

        pixels = zeros((num_rows, self.num_columns), uint8)

        row = 0
        for sample_data in super(TestSetIO, self).parse(file_path):
            pixels[row] = uint8(sample_data)
            row += 1

        return super(TestSetIO, self)._process_raw_pixel_values(pixels)

    def _check_headings(self, csv_reader):
        headings = csv_reader.next()
        super(TestSetIO, self)._check_heading_length(headings)


