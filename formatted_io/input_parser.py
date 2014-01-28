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

    def _parse_file_data_size(self, file_path):
        """Scans through the file to determine how many rows are being consumed

        Can be used to look ahead and precicely determine the size of data
        structures required

        @TODO: might be possible to reuse the csv reader object. Much effect on
                IO?
        """
        # -1 to account for the header
        size = -1
        with open(file_path, "rU") as sample:
            csv_reader = reader(sample)
            for row in csv_reader:
                size += 1
        return size


    def _process_raw_pixel_values(self, raw_pixels):
        raw_pixels /= 255
        return raw_pixels

    def _check_heading_length(self, headings):
        len_headings = len(headings)
        if (len_headings != self.num_columns):
            raise TypeError(
                "Expected to have %d columns. %d encountered" %
                (self.num_columns, len_headings)
            )


