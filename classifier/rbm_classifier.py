from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
import pdb


class RBMClassifier(object):
    """Classifies images as integer 0-9 and tests given input
    """

    # Choosing number of hidden components as 10 since we have 10 numbers to
    # learn
    num_components = 1

    classifier = None

    def train(self, target_numbers, pixels):
        classification_steps = [
            ("rbm", BernoulliRBM(n_components = self.num_components)),
            ("logistic", linear_model.LogisticRegression())
        ]
        self.classifier = Pipeline(steps=classification_steps)
        # pdb.set_trace()
        return self.classifier.fit(
            pixels,
            target_numbers
        )

    def predict(self, unclassified_image_pixels):
        if (self.classifier is None):
            raise RuntimeError("classification must occur before prediction")

        self.classifier.predict(unclassified_image_pixels)


