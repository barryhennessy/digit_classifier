from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from math import sqrt, ceil
import pdb

class RBMClassifier(object):
    """Classifies images as integer 0-9 and tests given input
    """

    # Choosing number of hidden components as 10 since we have 10 numbers to
    # learn
    num_components = 10

    classifier = None

    def train(self, target_numbers, pixels):
        rbm = BernoulliRBM(n_components = self.num_components)

        # @TODO: Figure out and tune these
        rbm.learning_rate = 0.06
        rbm.n_iter = 20

        logictic_regression = linear_model.LogisticRegression()

        # @TODO: Figure out and tune these
        # logictic_regression.C = 6000.0

        classification_steps = [
            ("rbm", rbm),
            ("logistic", logictic_regression)
        ]
        self.classifier = Pipeline(steps=classification_steps)

        return self.classifier.fit(
            pixels,
            target_numbers
        )

    def predict(self, unclassified_image_pixels):
        if (self.classifier is None):
            raise RuntimeError("classification must occur before prediction")

        return self.classifier.predict(unclassified_image_pixels)


    def plot_rbm_features(self):
        rbm = self.classifier.named_steps["rbm"]
        plt.figure(figsize=(4.2, 4))
        subplot_width = ceil(sqrt(self.num_components))
        for i, comp in enumerate(rbm.components_):
            plt.subplot(10, 10, i + 1)

            square_len = ceil(sqrt(comp.shape[0]))
            plt.imshow(comp.reshape((square_len, square_len)), cmap=plt.cm.gray_r,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('100 components extracted by RBM', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

        plt.show()