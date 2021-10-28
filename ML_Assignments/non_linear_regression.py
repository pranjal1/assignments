import numpy as np
from matplotlib import pyplot as plt


class NonLinearRegression:
    def __init__(self, train_x, train_y, lmbd=0):

        self.X = np.expand_dims(train_x, axis=1)
        self.PHI_X = self.get_phi_x(self.X)
        self.PHI_TILDE = self.PHI_X - np.mean(self.PHI_X, axis=0)
        self.Y = np.expand_dims(train_y, axis=1)
        self.Y_TILDE = self.Y - np.mean(self.Y)
        self.lmbd = lmbd

    def get_phi_x(self, X):
        X_sq = np.power(X, 2)
        X_cu = np.power(X, 3)
        X_fo = np.power(X, 4)
        return np.hstack([X, X_sq, X_cu, X_fo])

    def get_w_b_hat(self):
        self.w_hat = np.linalg.inv(
            self.PHI_TILDE.T @ self.PHI_TILDE
            + self.PHI_TILDE.shape[0] * self.lmbd * np.eye(self.PHI_TILDE.shape[1])
        ) @ (self.PHI_TILDE.T @ self.Y_TILDE)
        self.b_hat = np.mean(self.Y, axis=0) - self.w_hat.T @ np.mean(
            self.PHI_X, axis=0
        )

    def pred(self, X):
        X = np.expand_dims(X, axis=1)
        phi_x = self.get_phi_x(
            X
        ).T  # n phi_x's have dx1 shape, PHI_X on the other hand has shape dxn
        return self.w_hat.T @ phi_x + self.b_hat

    def get_error(self, X, Y):
        pred = self.pred(X)
        tgt = np.expand_dims(Y, axis=0)
        error = np.mean(np.power(tgt - pred, 2))
        return error


if __name__ == "__main__":
    np.random.seed(0)
    n = 10
    m = 8
    xtrain = np.linspace(0, 3, n)
    ytrain = -(xtrain ** 2) + 2 * xtrain + 2 + 0.5 * np.random.randn(n)
    xtest = np.linspace(0, 3, m)
    ytest = -(xtest ** 2) + 2 * xtest + 2 + 0.5 * np.random.randn(m)

    nlr = NonLinearRegression(xtrain, ytrain, lmbd=0.1)
    nlr.get_w_b_hat()
    print("W = {}".format(nlr.w_hat))
    print("B = {}".format(nlr.b_hat))
    ypred = nlr.pred(xtrain)
    print(nlr.get_error(xtrain, ytrain))

    plt.plot(xtrain, ytrain, "o")
    plt.plot(xtest, ytest, "x")
    plt.plot(xtest, -(xtest ** 2) + 2 * xtest + 2)
    plt.plot(xtrain, np.squeeze(ypred), "-")
    plt.legend(
        ["training samples", "test samples", "true line", "prediction (lambda = 0.1)"]
    )
    plt.show()

    results = []
    for lmbd in np.arange(0.001, 0.1, 0.001):
        nlr = NonLinearRegression(xtrain, ytrain, lmbd=lmbd)
        nlr.get_w_b_hat()
        training_error = nlr.get_error(xtrain, ytrain)
        testing_error = nlr.get_error(xtest, ytest)
        results.append([lmbd, training_error, testing_error])

    N = len(results)
    training_err = [x[1] for x in results]
    testing_err = [x[2] for x in results]
    lambdas = [x[0] for x in results]

    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(2, 1, 1)
    plt.ylabel("Error")
    plt.xlabel("Lambdas")
    plt.title("Training errors for different lambdas")
    plt.xticks(np.arange(min(lambdas), max(lambdas) + 0.001, 0.002), rotation=90)
    plt.plot(lambdas, training_err, "bo", label="Training Error")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.ylabel("Error")
    plt.xlabel("Lambdas")
    plt.title("Testing errors for different lambdas")
    plt.xticks(np.arange(min(lambdas), max(lambdas) + 0.001, 0.002), rotation=90)
    plt.plot(lambdas, testing_err, "r+", label="Testing Error")
    plt.legend()
    plt.show()
