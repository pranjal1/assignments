import numpy as np
from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, X, Y, lmbd=0):

        self.X = np.expand_dims(X, axis=1)
        self.X_tilde = self.X - np.mean(self.X)
        self.Y = np.expand_dims(Y, axis=1)
        self.Y_tilde = self.Y - np.mean(self.Y)

        self.theta = np.random.uniform(low=-0.1, high=0.1, size=(self.X.shape[0], 1))
        self.lmbd = lmbd

    def get_w_b_hat(self):
        self.w_hat = np.linalg.inv(
            self.X_tilde.T @ self.X_tilde
            + self.X_tilde.shape[1] * self.lmbd * np.eye(self.X_tilde.shape[1])
        ) @ (self.X_tilde.T @ self.Y_tilde)
        self.b_hat = np.mean(self.Y) - self.w_hat.T * np.mean(self.X)


if __name__ == "__main__":
    np.random.seed(0)
    n = 10
    x = np.linspace(0, 3, n)
    y = 2.0 * x + 1.0 + 0.5 * np.random.randn(n)
    y[9] = 20

    lr0 = LinearRegression(x, y, lmbd=0)
    lr1 = LinearRegression(x, y, lmbd=3)

    lr0.get_w_b_hat()
    lr1.get_w_b_hat()

    y_pred0 = lr0.w_hat * x + lr0.b_hat
    y_pred1 = lr1.w_hat * x + lr1.b_hat

    plt.plot(x, y, "o")
    plt.plot(x, 2 * x + 1)
    plt.plot(x, y_pred0.squeeze())
    plt.plot(x, y_pred1.squeeze())
    plt.legend(["data", "true line", "lambda=0", "lambda=3"])
    plt.show()
