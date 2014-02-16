from .rbm_classifier import RBMClassifier
from .random_forest_classifier import RandomForestClassifier
from .ada_boost_classifier import AdaBoostClassifier
from .classifier import Classifier


def get_classifier(classifier_name):
    """A factory method to get a valid classifier instance

    :raises KeyError: if the class requested doesn't exist in the module

    :param classifier_name: the name of the classifier to return

    :return: the requested classifier instance
    """

    # We're pulling in a reference to all the classes imported in this
    # *file* so we can reference classifier classes
    #
    # @TODO: this feels a bit dirty, but seems to be the best way
    symbols_avail_from_this_package = globals()

    classifier_class = symbols_avail_from_this_package[classifier_name]

    try:
        if issubclass(classifier_class, Classifier):
            classifier_instance = classifier_class()
            return classifier_instance
        else:
            raise RuntimeError(
                "The class requested was not a classifier: {0}".format(
                    classifier_name
                )
            )
    except TypeError:
        raise KeyError("Class '{0}' not found".format(classifier_class))
