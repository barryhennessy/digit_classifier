from csv import reader
from numpy import float32


class InputParser(object):
    """Parses kaggle input training/test set data for further processing

    See implementations for formats and implementation details

    @TODO: should probably be an abstract base class. Check out ABC module.
    """

    def parse(self, file_path):
        """Parses the test/training set and formats accordingly

        :param file_path: The path to be read from

        :return: A generator of lines in the csv file
        """
        with open(file_path, "rU") as sample:
            csv_reader = reader(sample)

            # Note: consumes the heading row
            self._check_headings(csv_reader)

            for sample_data in csv_reader:
                yield sample_data

    def _parse_file_data_size(self, file_path):
        """Scans through the file to determine how many rows are being consumed

        Can be used to look ahead and precisely determine the size of data
        structures required

        @TODO: might be possible to reuse the csv reader object. Much effect on
                IO?

        :param file_path: The path to read from

        :return: Integer. The number of lines in the file
        """
        # -1 to account for the header
        size = -1
        with open(file_path, "rU") as sample:
            csv_reader = reader(sample)
            for row in csv_reader:
                size += 1
        return size

    def _process_raw_pixel_values(self, raw_pixels):
        """Applies pre processing to the pixel values

        Pixel values are expected to be in the range 0-255.

        @TODO: raise an exception if the pixels are not in the correct range

        :param raw_pixels: A 2d numpy array of pixels (array of images) in the
                           range 0-255

        :return: A 2d numpy array of (float32) pixels in the range 0. - 1.
        """
        raw_pixels = float32(raw_pixels)
        raw_pixels /= 255
        return raw_pixels

    def _check_heading_length(self, headings):
        """Ensures that the headings are of the correct length as defined by
        the concrete implementation of the input parser

        :raises TypeError: If the headings are not the same length as expected

        :param headings: The headings of the file read

        """
        len_headings = len(headings)
        if len_headings != self.num_columns:
            raise TypeError(
                "Expected to have %d columns. %d encountered" %
                (self.num_columns, len_headings)
            )
