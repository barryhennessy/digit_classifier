from numpy import uint8, zeros
from csv import DictWriter
from formatted_io import InputParser


class TestSetIO(InputParser):
    """Parses kaggle input test set data for processing and formats predictions
    in the expected output format

    For training data input is expected in the form:

    pixel0, pixel1, ... pixelN
    22,33,...250
    53,0,...125
    ...

    Where the label column is absent for testing, the format is otherwise
    identical

    Pixels can range from 0 -> 255
    """

    num_columns = 784

    def parse(self, file_path):
        """Parses the test set for pixel values

        Generates arrays of pixel values

        :param file_path: The path to be read

        :return: Numpy array of images (uint8 arrays)
        """

        num_rows = self._parse_file_data_size(file_path)

        pixels = zeros((num_rows, self.num_columns), uint8)

        row = 0
        for sample_data in super(TestSetIO, self).parse(file_path):
            pixels[row] = uint8(sample_data)
            row += 1

        return super(TestSetIO, self)._process_raw_pixel_values(pixels)

    def write(self, predicted_values, path_or_file):
        """Writes the predicted values to the path specified

        :param predicted_values: A list of integer predictions, 0-9
        :param path_or_file:     A path or file object to write the formatted
                                 data to

        """
        try:
            with open(path_or_file, "w") as output_file:
                self._write_csv_formatted_predictions(output_file,
                                                      predicted_values)
        except TypeError:
            if isinstance(path_or_file, file):
                self._write_csv_formatted_predictions(path_or_file,
                                                      predicted_values)
            else:
                raise

    def _write_csv_formatted_predictions(self, output_file, predicted_values):
        """Writes formatted data to the file specified

        :param output_file:      The file object to write data to
        :param predicted_values: A list of values to write. In the range 0-9
        """
        csv_output = DictWriter(output_file, {"ImageId", "Label"})
        csv_output.writeheader()

        line_count = 1
        for value in predicted_values:
            row = {
                "ImageId": line_count,
                "Label": int(value)
            }
            csv_output.writerow(row)
            line_count += 1

    def _check_headings(self, csv_reader):
        """Consumes the first row of the csv_reader and ensures the headings
        are as required

        :param csv_reader: The CSV reader whose headings are to be checked
        """
        headings = csv_reader.next()
        super(TestSetIO, self)._check_heading_length(headings)
