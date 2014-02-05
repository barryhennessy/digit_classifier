__author__ = 'barryhennessy'

from classifier import Classifier
from sklearn.pipeline import Pipeline
import sklearn.ensemble


class AdaBoostClassifier(Classifier):
    """Classifies images to images using an AdaBoost approach"""

    default_n_estimators = 200

    default_learning_rate = 0.1

    default_algorithm = 'SAMME.R'

    def _get_classification_pipeline(self):
        """Builds and returns the classification Pipeline for this classifier

        :return: A Pipeline with the required classification steps
        """
        ada_boost = sklearn.ensemble.AdaBoostClassifier()

        ada_boost.n_estimators = self.default_n_estimators
        ada_boost.learning_rate = self.default_learning_rate
        ada_boost.algorithm = self.default_algorithm

        classification_steps = [
            ("ada_boost", ada_boost)
        ]

        return Pipeline(steps=classification_steps)
