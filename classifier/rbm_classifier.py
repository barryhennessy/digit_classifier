from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from classifier import Classifier


class RBMClassifier(Classifier):
    """Classifies images as integer 0-9 and tests given input
    """

    def _get_classification_pipeline(self):
        """Builds and returns the classification Pipeline for this classifier

        :return: A Pipeline with the required classification steps
        """
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

        return Pipeline(steps=classification_steps)
