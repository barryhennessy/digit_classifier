import unittest
from formatted_io import TestSetIO, TrainingSetIO
from numpy import ndarray, nditer

class TestRBMClassifier(unittest.TestCase):

    # @TODO: assumes testing from root directory
    MOCK_TRAINING_DATA_PATH="./test/formatted_io/mock_train_data.sm.csv"
    MOCK_TESTING_DATA_PATH="./test/formatted_io/mock_test_data.sm.csv"

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


if __name__ == '__main__':
    unittest.main()
