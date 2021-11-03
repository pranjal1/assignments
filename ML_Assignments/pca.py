import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import numpy as np


class PCA:
    def __init__(self, face_data_path: str) -> None:
        data = scipy.io.loadmat("yalefaces.mat")
        self.data = data["yalefaces"]

    def show_image(self, index: int = 50) -> None:
        # show nth image in the face dataset
        plt.imshow(self.data[:, :, index], cmap="gray")

    def calculate_pca(self):
        # calculate the eigen vectors and values from the covariance matrix
        vectorized_data = np.reshape(self.data, (2016, -1))
        mean_image = np.mean(vectorized_data, axis=1, keepdims=True)
        diff = vectorized_data - mean_image
        cov = np.cov(diff)
        self.values, self.vectors = np.linalg.eig(cov)

    def plot_eigen_values(self):
        # plot eigen values
        plt.figure(figsize=(20, 10))
        plt.xticks(np.arange(0, len(self.values), 50), rotation=45)
        plt.ylabel("Eigen Values")
        plt.xlabel("n")
        plt.scatter(
            y=self.values, x=list(range(1, len(self.values) + 1)), marker="o", c="r"
        )
        plt.show()

    def plot_eigen_contribution(self):
        # plot contribution of the eigen values in explaining the total variance
        percentage_variance = self.values / np.sum(self.values) * 100
        cumulative = np.cumsum(percentage_variance)
        plt.figure(figsize=(20, 10))
        plt.xticks(np.arange(0, len(self.values), 20), rotation=90)
        plt.ylabel(
            "Cumulative sum of percentage contribution of the first n Eigen Values"
        )
        plt.xlabel("n")
        plt.scatter(y=cumulative, x=list(range(len(self.values))))
        plt.show()

    def plot_top_n_eigen_faces(self, n=20):
        # plot top n eigen faces
        plt.figure(figsize=(20, 20))
        for i in range(1, n + 1):
            plt.subplot(4, 5, i)
            plt.imshow(self.vectors[:, i - 1].reshape(48, 42), cmap="gray")
        plt.show()


if __name__ == "__main__":
    p = PCA("yalefaces.mat")
    p.calculate_pca()
    p.plot_eigen_values()
    p.plot_eigen_contribution()
    p.plot_top_n_eigen_faces()
