from csv import reader
from pandas import SparseDataFrame

"""Parses kaggle input training/test set data for further processing

See implementations for formats and implementation details

@TODO: should probably be an abstract base class. Check out ABC module.
"""
class InputParser(object):
    def parse(self, file_path):
        """Parses the test/training set and formats accordingly
        """
        with open(file_path, "rU") as sample:
            csv_reader = reader(sample)

            # Note: consumes the heading row
            self._check_headings(csv_reader)

            for sample_data in csv_reader:
                yield sample_data

    def _process_raw_pixel_values(self, raw_pixels):
        pixels = SparseDataFrame(raw_pixels, default_fill_value=0)
        pixels /= 255
        return pixels

    def _check_heading_length(self, headings):
        len_headings = len(headings)
        if (len_headings != self.num_columns):
            raise TypeError(
                "Expected to have %d columns. %d encountered" %
                (self.num_columns, len_headings)
            )


