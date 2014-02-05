from sklearn.grid_search import GridSearchCV

__author__ = 'barryhennessy'


class Classifier(object):
    """Defines the basic structure of the classification objects

    All training and prediction is fundamentally the same: train the
    classifier on the target numbers and images and
    then attempt to classify new images
    """

    classifier = None

    def train(self, target_numbers, pixels, parameter_space=None):
        """Trains the classifier to recognise the images given as the
        associated numbers

        :param target_numbers:  An array of numbers corresponding to the
                                image in the pixels array
        :param pixels:          An array of pixel arrays (images)
                                corresponding to the target number
        :param parameter_space: Optional. A dict of value ranges for the
                                classifier to iterate through. Used to find
                                the optimal parameters for the classifier
        :return: None
        """
        pipeline = self._get_classification_pipeline()

        if parameter_space is not None:
            self.classifier = self._train_with_grid_search(
                pipeline,
                parameter_space
            )
        else:
            self.classifier = pipeline

        self.classifier.fit(
            pixels,
            target_numbers
        )

    def _train_with_grid_search(self, pipeline, parameter_space):
        """Sets up the classifier to be run multiple times with parameters
        in parameter_space to determine the optimal
        parameters

        :param pipeline:        The classification pipeline whose parameters
                                are being tuned
        :param parameter_space: A dict of value ranges for the classifier to
                                iterate through. Used to find the optimal
                                parameters for the classifier
        :return: An instance of GridSearchCV set up with the classifier and
                 parameter space
        """
        return GridSearchCV(
            pipeline,
            param_grid=parameter_space,
            verbose=2,
            n_jobs=3,
            pre_dispatch="10*n_jobs"
        )

    def print_grid_search_details(self, parameter_space):
        """Outputs information on the grid search and most successful
        parameters

        @TODO: Throw an error if grid search has not been done
        @TODO: Persist the parameter_space along with the GridSearchCV (is
        it already a property??)

        :param parameter_space: A dict of value ranges for the classifier to
                                iterate through. Used to find the optimal
                                parameters for the classifier
        :return: None
        """
        print("Best score: %0.3f" % self.classifier.best_score_)
        print("Best parameters set:")
        best_parameters = self.classifier.best_estimator_.get_params()

        for param_name in sorted(parameter_space.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    def predict(self, unclassified_image_pixels):
        """Predicts the number associated with the images given

        :param unclassified_image_pixels: An array of images (pixel arrays)
                                          to be classified

        :return: An array of integers (0-9) corresponding to the
                 classification of the image in the unclassified_image_pixels
                 array
        """
        if (self.classifier is None):
            raise RuntimeError("classification must occur before prediction")

        return self.classifier.predict(unclassified_image_pixels)
