import urllib.request
from collections import defaultdict
from typing import List, Union

import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split


def get_mean_and_covariance(X: np.ndarray, Y: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        X: Batch of Feature vector (nxd). n is the batch size and d is the dimension of each feature vector.
        Y: Label associated with each feature vector
    Returns:
        List of mean and coviarance tuples. Each tuple corresponds to unique label in Y.
    """
    # group x_train by labels
    x_grouped = [X[np.where(Y == l)] for l in np.unique(Y)]
    # calculate mean of each label group.
    means = [np.mean(x, axis=0).reshape(1, -1) for x in x_grouped]
    # subtract the data points from their respective label group mean
    diff = [x - m for x, m in zip(x_grouped, means)]
    # calculate covariance
    cov = np.zeros((X.shape[1], X.shape[1]))
    for diff_group in diff:
        for d in diff_group:
            cov += np.outer(d, d)
    cov /= Y.shape[0]
    # return mean and covariance
    return [(mean.squeeze(), cov) for mean in means]


def get_predictions(
    distributions: List[sp.stats._multivariate.multivariate_normal_frozen],
    qs: List[float],
    X: np.ndarray,
) -> List[int]:
    """
    Args:
        distributions: List of multivariate normal distributions P(X|Y)
        qs: List of marginal distribution Y. P(Y)
        X: Feature vector
    Returns:
        The distribution from the list of distributions in which X is most likely to belong.
    """
    # calculate the likelihood that the datapoint belong to each of the distributions
    pdfs = [[q * d.pdf(x) for q, d in zip(qs, distributions)] for x in X]
    # select the distribution with highest likelihood
    return [np.argmax(x) for x in pdfs]


def get_error(
    predictions: List[int], ground_truth: Union[List[int], np.ndarray]
) -> float:
    """
    Args:
        predictions: List of label predictions for each feature vector
        ground truth: List of ground truth label for each feature vector
    Returns:
        Error rate: F/(T+F)
    """
    accuracy = defaultdict(int)
    for pred, gt in zip(predictions, ground_truth):
        if pred == gt:
            accuracy["True"] += 1
        else:
            accuracy["False"] += 1
    return accuracy["False"] / len(predictions)


# Fetch the data and pre-process
url = " http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
x = dataset[:, 0:-1]
y = dataset[:, -1]

# split into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=17
)

# get mean and co-variance for each label group in train dataset
x_train_mean_cov = get_mean_and_covariance(x_train, y_train)
# get qs for each label group in training data
# p(x,y) = p(X=x|Y=y)*p(Y=y), q = P(Y=y)
qs = (np.unique(y_train, return_counts=True)[-1]) / len(y_train)

# initialize Gaussian distribution for each label group using the mean and co-variance
distributions = [
    sp.stats.multivariate_normal(*x, allow_singular=True) for x in x_train_mean_cov
]

# training performance
training_predictions = get_predictions(distributions, qs, x_train)
training_err = get_error(training_predictions, y_train)

# testing performance
testing_predictions = get_predictions(distributions, qs, x_test)
testing_err = get_error(testing_predictions, y_test)


print(f"The training error on {x_train.shape[0]} datapoints is {training_err:.4f}")
print(f"The testing error on {x_test.shape[0]} datapoints is {testing_err:.4f}")
