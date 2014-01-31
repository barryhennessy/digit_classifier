from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from math import sqrt, ceil

class RBMClassifier(object):
    """Classifies images as integer 0-9 and tests given input
    """

    classifier = None

    def train(self, target_numbers, pixels, parameter_space=None):
        rbm = BernoulliRBM()
        rbm.n_components = 100
        rbm.learning_rate = 0.01
        rbm.n_iter = 10

        logistic_regression = linear_model.LogisticRegression()
        logistic_regression.C = 10000

        classification_steps = [
            ("rbm", rbm),
            ("logistic", logistic_regression)
        ]

        pipeline = Pipeline(steps=classification_steps)

        if parameter_space is not None:
            self.classifier=self._train_with_grid_search(
                pipeline,
                parameter_space
            )
        else:
            self.classifier=pipeline

        self.classifier.fit(
            pixels,
            target_numbers
        )

    def _train_with_grid_search(self, pipeline, parameter_space):
        # @TODO: test this
        return GridSearchCV(
            pipeline,
            param_grid=parameter_space,
            verbose=2,
            n_jobs=3,
            pre_dispatch="10*n_jobs"
        )

    def print_grid_search_details(self, parameter_space):
        # @TODO: test this
        print("Best score: %0.3f" % self.classifier.best_score_)
        print("Best parameters set:")
        best_parameters = self.classifier.best_estimator_.get_params()
        for param_name in sorted(parameter_space.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))


    def predict(self, unclassified_image_pixels):
        if (self.classifier is None):
            raise RuntimeError("classification must occur before prediction")

        return self.classifier.predict(unclassified_image_pixels)


    def plot_rbm_features(self):
        rbm = self.classifier.named_steps["rbm"]
        plt.figure(figsize=(4.2, 4))
        subplot_width = ceil(sqrt(rbm.n_components))
        for i, comp in enumerate(rbm.components_):
            plt.subplot(subplot_width, subplot_width, i + 1)

            square_len = ceil(sqrt(comp.shape[0]))
            plt.imshow(comp.reshape((square_len, square_len)), cmap=plt.cm.gray_r,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('100 components extracted by RBM', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.show()