from numpy import uint8, float32, zeros
from formatted_io import InputParser


class TrainingSetIO(InputParser):
    """Parses kaggle input training and test set data for processing

    For training data input is expected in the form:

    label, pixel0, pixel1, ... pixelN
    1,22,33,...250
    9,53,0,...125
    ...

    Where the label column is absent for testing, the format is otherwise
    identical

    Pixels can range from 0 -> 255
    Labels can range from 0 -> 9
    """

    num_columns = 785

    def parse(self, file_path):
        """Parses the training set for labels (image numbers) and pixel values

        Generates tuples of integer labels and arrays of pixel values

        :param file_path: The path to be parsed

        :return: A tuple of (actual numbers, images). Where actual numbers is
                 an array of the integer value of the associated image. Images
                 is an array of an array of float32 pixels in the range 0. - 1.
        """

        num_rows = self._parse_file_data_size(file_path)

        pixels = zeros((num_rows, self.num_columns - 1), float32)
        numbers = zeros((num_rows, ), uint8)

        row = 0
        for sample_data in super(TrainingSetIO, self).parse(file_path):
            numbers[row] = uint8(sample_data[0])
            pixels[row] = uint8(sample_data[1:])
            row += 1

        pixels = super(TrainingSetIO, self)._process_raw_pixel_values(pixels)

        return (numbers, pixels)

    def _check_headings(self, csv_reader):
        """Consumes the first row of the csv_reader and ensures the headings
        are long enough and have the label field as the first position

        :throws TypeError: If the file doesn't have the label header or it is
                           not where expected

        :param csv_reader: A CSV_reader whose headings are to be checked
        """
        headings = csv_reader.next()

        if (headings[0] != "label"):
            raise TypeError(
                "Training set must have a label field as it's first value"
            )

        super(TrainingSetIO, self)._check_heading_length(headings)
