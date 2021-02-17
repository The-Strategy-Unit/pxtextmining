#############################################################################
# Upbalancing
# ------------------------------------
# First, create a function that detects rare classes, i.e. classes whose number of records
# is smaller than the specified threshold. The function returns a dictionary. The keys are the rare classes and
# the values are the user-specified up-balancing numbers for each class (can be numeric or array of numeric).
# Second, define a function that uses the first function to return the up-balanced dataset.
# Third, pass the second function to imblearn.FunctionSampler to be then passed to the pipeline (see script where
# pipeline is constructed).

# This function finds classes with counts less than a specified threshold and up-balances them based on
# user-specified number(s). The function performs a few validity checks:
# 1. The threshold must be smaller than the up-balancing number(s). When it's not,
#     the latter takes the value of the former.
# 2. When the up-balancing number is zero or the threshold is smaller than all class counts, then the function
#     returns the original counts.
# The validity checks ensure that the function does not stop the script. It's completely the user's responsibility
# to ensure that the supplied values are meaningful. For example, if each of the rare classes are > 200 in number but
# the threshold were 100 in one run and 150 in another run of the pipeline, then the result would be the original counts
# in both cases, i.e. there would be a redundant repetition of runs.
# Finally, the up-balancing number can be 0, an integer or a list of integers with length = number of rare classes.
# It's the user's responsibility to ensure that, when it's a list, it has the correct length.
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def random_over_sampler_dictionary(y, threshold=200, up_balancing_counts=300):
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
    aux = random_over_sampler_dictionary(y, threshold, up_balancing_counts)
    return RandomOverSampler(
        sampling_strategy=aux,
        random_state=random_state).fit_resample(X, y)
