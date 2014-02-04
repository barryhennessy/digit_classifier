import matplotlib.pyplot as plt
from math import sqrt, ceil
from numpy import uint8, float32, float64, asarray

__author__ = 'barryhennessy'


class ImageDisplay(object):
    """Handles all plotting and visual analysis of test/train images
    """

    def plot_incorrect_classifications(self, predicted_numbers, actual_numbers, images):
        """Plots misclassified numbers in order of value.

        Images will be plotted in order, i.e. all 1's first, 2's second, etc.

        :param predicted_numbers: The numbers predicted from the training data
        :param actual_numbers: The actual numbers from the training data
        :param images: The images from the training data
        :return: Pyplot object with the misclassified numbers plotted on it
        """
        # Is a priority queue what I need here?
        numbers_to_plot = {number: [] for number in range(0, 10, 1)}
        for index, actual_number in enumerate(actual_numbers):
            if predicted_numbers[index] != actual_number:
                numbers_to_plot[actual_number].append(images[index])


        # Sorting numbers so they're plotted in order
        plot_numbers = []
        for number in range(0, 10, 1):
            plot_numbers.extend(numbers_to_plot[number][1:])
        plot_images = asarray(plot_numbers, dtype="float32")

        return self.plot_images(plot_images, (10, 10))


    def plot_images(self, images, figure_size=(4.2, 4)):
        """Plots the images given on a square grid

        Images are to be in the range 0.0-1.0
        :rtype : pyplot object
        """
        plt.figure(figsize=figure_size)

        num_images_per_line = float(ceil(sqrt(len(images))))

        # Converting images back from 0-1 floats to 0-255 integer format
        images *= 255
        images = uint8(images)

        for i, pixels in enumerate(images):
            plt.subplot(num_images_per_line, num_images_per_line, i + 1)

            square_len = ceil(sqrt(len(pixels)))
            image_pixels = pixels.reshape((square_len, square_len))

            plt.imshow(image_pixels, cmap=plt.cm.gray_r, interpolation='nearest')

            plt.xticks(())
            plt.yticks(())

        title = '{0} images'.format(len(images))
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        return plt
