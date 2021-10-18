import random
from tqdm import tqdm

import scipy.io
import numpy as np
from matplotlib import pyplot as plt


class SoftMarginClassifier:
    def __init__(
        self, data_path: str, theta_update_threshold: float = 0.001, C=100, lr=0.01
    ) -> None:
        """
        Args:
            data_path: path to the mnist_49_3000.mat file
            theta_update_threshold: Continue converging if update in theta is mopre than ``theta_update_threshold``
        """
        # read data from file
        data = scipy.io.loadmat(data_path)
        x = np.array(data["x"])
        y = np.array(data["y"][0])
        y = np.expand_dims(y, 0)

        self.C = C
        self.lr = lr

        # test-train split
        self.training_x, self.testing_x = x[:, :2000], x[:, 2000:]
        self.training_y, self.testing_y = y[:, :2000].T, y[:, 2000:].T

        # initialize theta randomly
        self.W = np.random.uniform(
            low=-0.1, high=0.1, size=(self.training_x.shape[0], 1)
        )
        self.B = 0.0
        self.theta_update_threshold = theta_update_threshold

    def get_image(
        self, index: int, from_training_data: bool = True, plot: bool = True
    ) -> np.ndarray:
        """
        Class method to convert flattened data to image
        """
        # change the index to show different images
        dataset = self.training_x if from_training_data else self.testing_x
        image = dataset[:, index].reshape(28, 28)
        if plot:
            plt.imshow(image, interpolation="nearest")
            plt.show()
        return image

    def get_h(self, X):
        return -1 if X < 1 else 0

    def get_dJdb(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        This will compute the gradient of the objective function  wrt b
        """
        dbs = []
        for i in range(X.shape[1]):
            xi = X[:, i : i + 1]
            yi = Y[i : i + 1]
            dbs.append(yi * (self.get_h((self.W.T @ xi + self.B) * yi)))
        db = np.mean(dbs) * self.C
        return db

    def get_dJdw(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        This will compute the gradient of the objective function  wrt w
        """
        dws = []
        for i in range(X.shape[1]):
            xi = X[:, i : i + 1]
            yi = Y[i : i + 1]
            dws.append(xi * yi * (self.get_h((self.W.T @ xi + self.B) * yi)))
        dw = np.mean(dws, axis=0) * self.C + self.W
        return dw

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
                    new_W = self.W - self.lr * self.get_dJdw(X, Y)
                    new_B = self.B - self.lr * self.get_dJdb(X, Y)
                    update_in_W = np.mean(np.abs(self.W - new_W))
                    update_in_B = np.mean(np.abs(self.B - new_B))
                    if update_in_W + update_in_B < self.theta_update_threshold:
                        return
                    self.W = new_W
                    self.B = new_B
        except KeyboardInterrupt:
            pass

    def get_testing_error(self):
        """
        This will get the testing error i.e. ratio of the incorrect predictions to the total testing samples
        It gets the predictions using the theta at optimum
        """
        self.test_prediction_score = self.W.T @ self.testing_x + self.B
        test_prediction = np.where(
            self.test_prediction_score >= 0, 1, self.test_prediction_score
        )
        test_prediction = np.where(test_prediction < 0, -1, test_prediction)
        test_prediction = test_prediction.reshape(-1, 1)
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
        self.test_dist_from_boundary = np.abs(self.test_prediction_score) / (
            np.sqrt(np.sum(np.square(self.W)))
        )
        bad_index = np.where(self.test_result == False)[0]
        bad_results = [(i, self.test_dist_from_boundary[0, i]) for i in bad_index]
        bad_results_sorted = sorted(bad_results, key=lambda x: x[1], reverse=True)
        top_num_bad_res = [i for i, _ in bad_results_sorted[:num]]
        bad_images = [self.get_image(i, False, False) for i in top_num_bad_res]

        f, axarr = plt.subplots(1, num)

        for i in range(len(bad_images)):
            axarr[i].imshow(bad_images[i])
        plt.show(block=True)

    def objective_fn(self):
        second_terms = []
        for i in range(self.testing_x.shape[1]):
            xi = self.testing_x[:, i : i + 1]
            yi = self.testing_y[i : i + 1]
            second_terms.append(np.max([0, np.squeeze((self.W.T @ xi + self.B) * yi)]))
        second_term = np.sum(second_terms) * self.C / (self.testing_x.shape[1])
        first_term = 0.5 * (np.sum(np.square(self.W)))
        print(
            "The value of objective function at optimum is {}".format(
                first_term + second_term
            )
        )


if __name__ == "__main__":
    data_path = "mnist_49_3000.mat"
    log_classifier = SoftMarginClassifier(
        data_path=data_path, theta_update_threshold=0.001
    )
    log_classifier.converge()
    log_classifier.get_testing_error()
    print(
        "The termination criterion is: the mean of absolute in W and B in the new iteration is less that 0.002."
    )
    log_classifier.objective_fn()
    log_classifier.plot_bad_results()
