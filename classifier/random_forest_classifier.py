from classifier import Classifier
from sklearn.pipeline import Pipeline
import sklearn.ensemble

__author__ = 'barryhennessy'


class RandomForestClassifier(Classifier):
    """Classifies images to images using a random forest classification approach
    """

    default_max_depth = 20
    default_max_features = 'auto'
    default_min_samples_leaf = 1
    default_min_samples_split = 2
    default_n_estimators = 50


    def _get_classification_pipeline(self):
        """Builds and returns the classification Pipeline for this classifier

            :return: A Pipeline with the required classification steps
            """
        forest = sklearn.ensemble.RandomForestClassifier()

        forest.max_depth = self.default_max_depth
        forest.max_features = self.default_max_features
        forest.min_samples_leaf = self.default_min_samples_leaf
        forest.min_samples_split = self.default_min_samples_split
        forest.n_estimators = self.default_n_estimators

        classification_steps = [
            ("random_forest", forest)
        ]

        return Pipeline(steps=classification_steps)
