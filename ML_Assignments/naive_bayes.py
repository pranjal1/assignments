"""
Naive Bayes Implementation
Author: Pranjal Dhakal
"""

from typing import Union

import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split


def get_P_Y(Y: np.ndarray) -> dict:
    # computes the prior probability P(Y)
    # returns dictionary with prior probability assocaited with each unique label
    unique_labels, counts = np.unique(Y, return_counts=True)
    return {k: v for k, v in zip(unique_labels, counts / Y.shape[0])}


def get_P_X_given_Y(
    observation: np.ndarray,
    label: Union[float, int],
    X: np.ndarray,
    Y: np.ndarray,
    L: int,
) -> float:
    # computes P(X | Y)
    # the probability is the product of probabilities at each feature dimension

    # take only relevant Xs in training using matching label
    relevant_X = X[np.where(Y == label)]
    # At each feature dimension, find the matches with the observation
    # difference shoud be 0 if feature at a dimension matches with observation
    relevant_X -= observation
    matching_xis = np.sum(relevant_X == 0, axis=0)
    # calculate probability, add 1 and L in num and denum according to formula
    gis = (matching_xis + 1) / (relevant_X.shape[0] + L)
    # multiply the individual probabilities
    return np.product(gis, axis=0)


if __name__ == "__main__":
    url = " http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    raw_data = urllib.request.urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",")
    x = dataset[:, 0:-1]
    m = np.median(x, axis=0)
    x = (x > m) * 2 + (x <= m) * 1
    # making the feature vectors binary
    y = dataset[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=17
    )

    train_P_Y = get_P_Y(y_train)
    unique_labels = list(train_P_Y.keys())

    predictions = []

    for test_observation in x_test:
        # for each observation in test dataset, compute the probability of it belonging to different labels
        P_labels = [
            [
                label,
                train_P_Y[label]
                * get_P_X_given_Y(
                    test_observation, label, x_train, y_train, len(unique_labels)
                ),
            ]
            for label in unique_labels
        ]
        # assign the label whose probability is maximum
        predictions.append(max(P_labels, key=lambda x: x[-1])[0])
    result = np.unique(np.equal(y_test, np.array(predictions)), return_counts=True)
    result = {k: v for k, v in zip(*result)}
    print("test accuracy = {}".format(result[True] / y_test.shape[0]))
    print("test error = {}".format(result[False] / y_test.shape[0]))
