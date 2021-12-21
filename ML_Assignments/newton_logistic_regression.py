import random
from tqdm import tqdm

import scipy.io
import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression:
    def __init__(self, data_path: str, theta_update_threshold: float = 0.001) -> None:
        """
        Args:
            data_path: path to the mnist_49_3000.mat file
            theta_update_threshold: Continue converging if update in theta is mopre than ``theta_update_threshold``
        """
        # read data from file
        data = scipy.io.loadmat(data_path)
        x = np.array(data["x"])
        y = np.array(data["y"][0])

        y[y == -1] = 0
        y = np.expand_dims(y, axis=0)

        # test-train split
        self.training_x, self.testing_x = x[:, :2000], x[:, 2000:]
        self.training_y, self.testing_y = y[:, :2000], y[:, 2000:]

        # get x_tilde for training and testing x
        self.training_x = self.get_x_tilde(self.training_x)
        self.testing_x = self.get_x_tilde(self.testing_x)

        # initialize theta randomly
        self.theta = np.random.uniform(
            low=-0.1, high=0.1, size=(self.training_x.shape[0], 1)
        )

        # lambda value is set to 10
        self.lmbd = 10
        self.theta_update_threshold = theta_update_threshold

    def get_image(
        self, index: int, from_training_data: bool = True, plot: bool = True
    ) -> np.ndarray:
        """
        Class method to convert flattened data to image
        """
        # change the index to show different images
        dataset = self.training_x if from_training_data else self.testing_x
        image = dataset[:, index][1:].reshape(28, 28)
        if plot:
            plt.imshow(image, interpolation="nearest")
            plt.show()
        return image

    def get_x_tilde(self, X: np.ndarray) -> None:
        """
        Method to get X_tilde from X. Just append 1 to the top of the X vector
        """
        return np.vstack([np.ones((1, X.shape[1])), X])

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        """
        Returns sigmoid activation of the given input
        """
        return 1 / (1 + np.exp(-X))

    def get_DDJ(self, X: np.ndarray) -> np.ndarray:
        """
        This will compute the Hessian of objective function
        """
        sigma_thetaT_x = self.sigmoid(self.theta.T @ X)
        prod = sigma_thetaT_x * (1 - sigma_thetaT_x)
        ddj = (X @ np.diag(prod[0]) @ X.T) + 2 * self.lmbd * np.eye(X.shape[0])
        return ddj

    def get_DJ(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        This will compute the gradient of the objective function
        """
        dj = X * (self.sigmoid(self.theta.T @ X) - Y)
        dj = np.sum(dj, axis=1, keepdims=True) + 2 * self.lmbd * self.theta
        return dj

    def objective_fn(self) -> None:
        """
        This will compute and print the value of the log-objective function J at optimum
        """
        X, Y = self.training_x, self.training_y
        inside_summation = Y * np.log(self.sigmoid(self.theta.T @ X)) + (
            1 - Y
        ) * np.log(1 - self.sigmoid(self.theta.T @ X))
        summation = np.sum(inside_summation, axis=1)
        print("Value of objective function at optimum = {}".format(summation.item()))

    def converge(self) -> None:
        """
        This will iterate until the termination criteria is met i.e. update_in_theta < theta_update_threshold
        update in theta is computed as np.mean(np.abs(self.theta - new_theta))
        """
        try:
            with tqdm() as pbar:
                while 1:
                    pbar.update(1)
                    X, Y = self.training_x, self.training_y
                    new_theta = self.theta - np.linalg.inv(
                        self.get_DDJ(X)
                    ) @ self.get_DJ(X, Y)
                    update_in_theta = np.mean(np.abs(self.theta - new_theta))
                    if update_in_theta < self.theta_update_threshold:
                        return
                    self.theta = new_theta
        except KeyboardInterrupt:
            pass

    def get_testing_error(self):
        """
        This will get the testing error i.e. ratio of the incorrect predictions to the total testing samples
        It gets the predictions using the theta at optimum
        """
        self.test_prediction_confidence = self.sigmoid(self.theta.T @ self.testing_x)
        test_prediction = 1 * (self.test_prediction_confidence > 0.5)
        self.test_result = self.testing_y == test_prediction
        result = np.unique(self.test_result, return_counts=True)
        for res, counts in zip(*result):
            if res != True:
                err = (counts / self.testing_x.shape[1]) * 100
        print(
            "Test error on {} samples is {}".format(
                self.testing_x.shape[1], np.around(err, 3)
            )
        )

    def plot_bad_results(self, num: int = 5) -> None:
        """
        This will plot top-n bad prediction.
        The top bad predictions are assigned to those incorrect predictions thata the classifier was most confident about.
        When the classifier predicts value close to 0 or 1, it is most confident as compared to 0.5 when it is most confused or unsure.
        So I used the predictions close to 0 or 1 that are incorrect.
        Then I plotted those results.
        """
        bad_index = np.where(self.test_result == False)[1]
        bad_results = [
            (i, np.abs(self.test_prediction_confidence[0, i] - 0.5)) for i in bad_index
        ]
        bad_results_sorted = sorted(bad_results, key=lambda x: x[1], reverse=True)
        top_num_bad_res = [i for i, _ in bad_results_sorted[:num]]
        print([self.test_prediction_confidence[0, i] for i in top_num_bad_res])
        bad_images = [self.get_image(i, False, False) for i in top_num_bad_res]

        plt.figure()
        f, axarr = plt.subplots(1, num)

        for i in range(len(bad_images)):
            axarr[i].imshow(bad_images[i])
        plt.show(block=True)


if __name__ == "__main__":
    data_path = "mnist_49_3000.mat"
    log_classifier = LogisticRegression(
        data_path=data_path, theta_update_threshold=0.001
    )
    log_classifier.converge()
    log_classifier.get_testing_error()
    print(
        "The termination criterion is: the mean of absolute update in theta in the new iteration is less that 0.001."
    )
    log_classifier.objective_fn()
    log_classifier.plot_bad_results()
