import unittest
import os
from formatted_io import TestSetIO, TrainingSetIO
from numpy import ndarray, nditer, dtype
from csv import reader

class TestRBMClassifier(unittest.TestCase):

    # @TODO: assumes testing from root directory
    MOCK_TRAINING_DATA_PATH="./test/formatted_io/mock_train_data.sm.csv"
    MOCK_TESTING_DATA_PATH="./test/formatted_io/mock_test_data.sm.csv"

    MOCK_TESTING_WRITE_PATH="./test/formatted_io/mock_test_write.csv"

    def setUp(self):
        if os.path.exists(self.MOCK_TESTING_WRITE_PATH):
            os.remove(self.MOCK_TESTING_WRITE_PATH)

    # @TODO: test improper headings throwing an exception

    def test_training_data_format(self):
        """Tests that the training data is of the excepted format
        ([numbers], [pixels]) with the same number of numbers to pixels
        """
        input_parser = TrainingSetIO()
        training_numbers, training_pixels = input_parser.parse(
            self.MOCK_TRAINING_DATA_PATH
        )
        
        self.assertEqual(len(training_numbers), len(training_pixels))

    def test_training_excepts_with_wrong_headings(self):
        """Tests that the proper exception gets thrown if the wrong headings
        are encountered
        """
        input_parser = TrainingSetIO()
        with self.assertRaises(TypeError):
            training_numbers, training_pixels = input_parser.parse(
                # Wrong path
                self.MOCK_TESTING_DATA_PATH
            )

    def test_training_pixels_sparse_format(self):
        """Tests that the training data is of the correct number array format
        """
        input_parser = TrainingSetIO()
        training_numbers, training_pixels = input_parser.parse(
            self.MOCK_TRAINING_DATA_PATH
        )
        
        self.assertIsInstance(training_pixels, ndarray)


    def test_training_numbers_sparse_format(self):
        """Tests that the training data is of the correct number array format
        """
        input_parser = TrainingSetIO()
        training_numbers, training_pixels = input_parser.parse(
            self.MOCK_TRAINING_DATA_PATH
        )
        
        self.assertIsInstance(training_numbers, ndarray)

    def test_training_data_pixel_normalising(self):
        """Tests that the training data pixels have been normalised to 0-1
        scale
        """
        input_parser = TrainingSetIO()
        training_numbers, training_pixels = input_parser.parse(
            self.MOCK_TRAINING_DATA_PATH
        )
        for training_pixel in nditer(training_pixels):
            self.assertGreaterEqual(training_pixel, 0)
            self.assertLessEqual(training_pixel, 1)
            self.assertIs(training_pixel.dtype, dtype("float32"))

    def test_training_data_numbers_correct_range(self):
        """Tests that the training data numbers are in the range 0-9
        """
        input_parser = TrainingSetIO()
        training_numbers, training_pixels = input_parser.parse(
            self.MOCK_TRAINING_DATA_PATH
        )

        number_range = range(0, 10, 1)
        for training_number in nditer(training_numbers):
            self.assertIn(training_number, number_range)
     
    def test_testing_excepts_with_wrong_headings(self):
        """Tests that the proper exception gets thrown if the wrong headings
        are encountered
        """
        input_parser = TestSetIO()
        with self.assertRaises(TypeError):
            training_numbers, training_pixels = input_parser.parse(
                # Wrong path
                self.MOCK_TRAINING_DATA_PATH
            )   

    def test_testing_data_sparse_format(self):
        """Tests that the testing data is of the correct number array format
        """
        input_parser = TestSetIO()
        testing_pixels = input_parser.parse(self.MOCK_TESTING_DATA_PATH)
        self.assertIsInstance(testing_pixels, ndarray)


    def test_testing_data_pixel_normalising(self):
        """Tests that the testing data pixels have been normalised to the range
        0-1
        """
        input_parser = TestSetIO()
        testing_pixels = input_parser.parse(self.MOCK_TESTING_DATA_PATH)

        for training_pixel in nditer(testing_pixels):
            self.assertGreaterEqual(training_pixel, 0)
            self.assertLessEqual(training_pixel, 1)

    def test_output_format_raises_on_bad_values(self):
        predicted_values = [0, 1, 2, "foo", 4, 5]
        IO = TestSetIO()
        with self.assertRaises(ValueError):
            IO.write(predicted_values, self.MOCK_TESTING_WRITE_PATH)

    def test_output_format_raises_on_non_writable_file(self):
        predicted_values = range(0, 10, 1)
        IO = TestSetIO()
        # @TODO: check if the file lingers after this test
        with self.assertRaises(IOError):
            IO.write(predicted_values, "bad/path")

    def test_output_file_exists_after_write(self):
        # Ensuring that no leaky tests leave files around that would cause this
        # to break
        self.assertEqual(os.path.exists(self.MOCK_TESTING_WRITE_PATH), False)

        predicted_values = range(0, 10, 1)
        IO = TestSetIO()
        IO.write(predicted_values, self.MOCK_TESTING_WRITE_PATH)

        self.assertEqual(os.path.exists(self.MOCK_TESTING_WRITE_PATH), True)

    def test_output_file_format(self):
        predicted_values = range(0, 10, 1)
        IO = TestSetIO()
        IO.write(predicted_values, self.MOCK_TESTING_WRITE_PATH)

        with open(self.MOCK_TESTING_WRITE_PATH, "rU") as sample:
            csv_reader = reader(sample)

            headings = csv_reader.next()
            self.assertEqual(headings[0], "ImageId")
            self.assertEqual(headings[1], "Label")

            for written_line in csv_reader:
                # output indices are indexed at 1
                prediction_index = int(written_line[0]) - 1
                predicted_value = int(written_line[1])

                self.assertEqual(
                    predicted_value,
                    predicted_values[prediction_index]
                )

if __name__ == '__main__':
    unittest.main()
