import random
from tqdm import tqdm

import scipy.io
import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression:
    def __init__(self, data_path: str, theta_update_threshold: float = 0.001) -> None:
        data = scipy.io.loadmat(data_path)
        x = np.array(data["x"])
        y = np.array(data["y"][0])

        y[y == -1] = 0
        y = np.expand_dims(y, axis=0)
        self.training_x, self.testing_x = x[:, :2000], x[:, 2000:]
        self.training_y, self.testing_y = y[:, :2000], y[:, 2000:]

        self.training_x = self.get_x_tilde(self.training_x)
        self.testing_x = self.get_x_tilde(self.testing_x)

        self.theta = np.random.uniform(
            low=-0.1, high=0.1, size=(self.training_x.shape[0], 1)
        )
        self.lmbd = 10
        self.theta_update_threshold = theta_update_threshold

    def get_image(
        self, index: int, from_training_data: bool = True, plot: bool = True
    ) -> np.ndarray:
        # change the index to show different images
        dataset = self.training_x if from_training_data else self.testing_x
        image = dataset[:, index][1:].reshape(28, 28)
        if plot:
            plt.imshow(image, interpolation="nearest")
            plt.show()
        return image

    def get_x_tilde(self, X: np.ndarray) -> None:
        return np.vstack([np.ones((1, X.shape[1])), X])

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def get_DDJ(self, X: np.ndarray) -> np.ndarray:
        sigma_thetaT_x = self.sigmoid(self.theta.T @ X)
        prod = sigma_thetaT_x * (1 - sigma_thetaT_x)
        ddj = (X @ np.diag(prod[0]) @ X.T) + 2 * self.lmbd * np.eye(X.shape[0])
        return ddj

    def get_DJ(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        dj = X * (self.sigmoid(self.theta.T @ X) - Y)
        dj = np.sum(dj, axis=1, keepdims=True) + 2 * self.lmbd * self.theta
        return dj

    def objective_fn(self) -> None:
        X, Y = self.training_x, self.training_y
        inside_summation = Y * np.log(self.sigmoid(self.theta.T @ X)) + (
            1 - Y
        ) * np.log(1 - self.sigmoid(self.theta.T @ X))
        summation = np.sum(inside_summation, axis=1)
        print("Value of objective function at optimum = {}".format(summation.item()))

    def converge(self) -> None:
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

    def get_prediction(self, X: np.ndarray) -> np.ndarray:
        return 1 * (self.sigmoid(self.theta.T @ X) > 0.5)

    def get_testing_error(self):
        test_prediction = self.get_prediction(self.testing_x)
        self.test_result = self.testing_y == test_prediction
        result = np.unique(self.test_result, return_counts=True)
        for res, counts in zip(*result):
            if res != True:
                err = counts / self.testing_x.shape[1]
        print(
            "Test error on {} samples is {}".format(
                self.testing_x.shape[1], np.around(err, 3)
            )
        )

    def plot_bad_results(self, num: int = 5) -> None:
        bad_index = np.where(self.test_result == False)[1]
        n_random_bad_index = random.sample(list(bad_index), num)
        bad_images = [self.get_image(i, False, False) for i in n_random_bad_index]
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
