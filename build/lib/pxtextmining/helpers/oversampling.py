import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def random_over_sampler_dictionary(y, threshold=200, up_balancing_counts=300):
    """
    Function that detects rare classes.

    Finds classes with counts fewer than a specified threshold. The function performs a few validity checks:

    1. The threshold must be smaller than the up-balancing number(s). When it is not,
    the latter takes the value of the former.

    2. When the up-balancing number is zero or the threshold is smaller than all class counts, then the function
    returns the original counts.

    The validity checks ensure that the function does not stop the script. It is completely the user's responsibility
    to ensure that the supplied values are meaningful. For example, if each of the rare classes are > 200 in number but
    the threshold were 100 in one run and 150 in another run of the pipeline, then the result would be the original
    counts in both cases, i.e. there would be a redundant repetition of runs.
    Finally, the up-balancing number can be 0, an integer or a list of integers with length = number of rare classes.
    It is the user's responsibility to ensure that, when it is a list, it has the correct length.

    :param ndarray y: The dependent variable. Shape (n_samples, ).
    :param int threshold: The class count below which a class is considered rare.
    :param array[int] up_balancing_counts: The number by which to up-balance a class.
    :return: rare_classes (`dict`): Keys are the rare classes and values are the user-specified up-balancing numbers for
        each class.
    """

    unique, frequency = np.unique(y, return_counts=True)
    rare_classes = pd.DataFrame()
    rare_classes['counts'], rare_classes.index = frequency, unique

    if type(up_balancing_counts) is int:
        up_balancing_counts = [up_balancing_counts]

    aux = list(filter(lambda x: up_balancing_counts[x] < threshold,
                      range(len(up_balancing_counts))))
    if any(x < threshold for x in up_balancing_counts):
        for i in aux:
            print("The supplied up-balancing value for class " +
                  rare_classes.index[aux] +
                  " is smaller than the supplied threshold value. "
                  "Setting up_balancing_counts = threshold for this class")
            up_balancing_counts[i] = threshold

    if (len(rare_classes[rare_classes.counts < threshold]) == 0) or (up_balancing_counts == [0]):
        rare_classes = rare_classes.to_dict()['counts']
    else:
        rare_classes = rare_classes[rare_classes.counts < threshold]

        if len(up_balancing_counts) != 1:
            rare_classes.counts = up_balancing_counts
        else:
            rare_classes.counts = up_balancing_counts * len(rare_classes.counts)
        rare_classes = rare_classes.to_dict()['counts']
    return rare_classes


def random_over_sampler_data_generator(X, y, threshold=200, up_balancing_counts=300, random_state=0):
    """
    Uses random_over_sampler_dictionary() to return the up-balanced dataset.
    Can be passed to imblearn.FunctionSampler to be then passed to imblearn.pipeline.

    :param ndarray X: The features table. Shape (n_samples, n_features)
    :param ndarray y: The dependent variable. Shape (n_samples, ).
    :param int threshold: The class count below which a class is considered rare.
    :param array[int] up_balancing_counts: The number by which to up-balance a class.
    :param int random_state: RandomState instance or ``None``, optional (default=``None``).
    :return: self.
    """

    aux = random_over_sampler_dictionary(y, threshold, up_balancing_counts)
    return RandomOverSampler(
        sampling_strategy=aux,
        random_state=random_state).fit_resample(X, y)
