from unittest import TestCase
from classifier import get_classifier, RandomForestClassifier

__author__ = 'barryhennessy'

class TestFactory(TestCase):
    """Tests for the classifier factory function"""

    def test_invalid_class_raises_key_error(self):
        with self.assertRaises(KeyError):
            classifier = get_classifier("invalid_class")

    def test_valid_class_gives_expected(self):
        classifier = get_classifier("RandomForestClassifier")
        self.assertIsInstance(classifier, RandomForestClassifier)

    def test_invalid_spelling_raises_key_error(self):
        with self.assertRaises(KeyError):
            classifier = get_classifier("random_forestClassifier")