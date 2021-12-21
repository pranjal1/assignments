import csv
import random

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


class EM:
    def __init__(self, csv_path):
        self._data = csv.reader(
            open(csv_path), delimiter=",", quoting=csv.QUOTE_NONNUMERIC
        )
        self.data = np.array(list(self._data))  # every row is a x
        self.N = self.data.shape[0]
        self.D = self.data.shape[1]
        self.K = 3

    def viz_data(self, with_mean=False):
        plt.scatter(self.data[:, 0], self.data[:, 1], marker="x")
        if with_mean:
            final_mu = self.result[1]
            for i in range(self.K):
                plt.scatter(final_mu[i][0], final_mu[i][1], marker="o", color="red")
        plt.show()

    def initialize_params(self):
        phi_j = np.full((self.N, self.K), 1 / self.K)
        mu_j = self.data[random.sample(range(self.N), self.K)]
        cov_j = [np.cov(self.data.T)] * self.K
        return (phi_j, mu_j, cov_j)

    def _cov_with_weight(self, X, mu_j, w_ij):
        new_covs = []
        for i in range(self.K):
            X_minus_xbar = X - mu_j[i]
            numerator = np.einsum("ij,ik->ijk", X_minus_xbar, X_minus_xbar)
            w_col = w_ij[:, i : i + 1]
            numerator = np.sum(numerator * np.expand_dims(w_col, axis=1), axis=0)
            denum = np.sum(w_col)
            new_covs.append(numerator / denum)
        return new_covs

    def m_step(self, w_ij, mu_j):
        new_mus = []
        new_phis = []
        for i in range(self.K):
            w_col = w_ij[:, i : i + 1]
            numerator = np.sum(self.data * w_col, axis=0)
            denum = np.sum(w_col)
            numerator /= denum
            new_mus.append(numerator)
            new_phis.append(denum / self.N)
        new_phis = [new_phis] * self.N
        new_covs = self._cov_with_weight(self.data, mu_j, w_ij)
        return (new_phis, new_mus, new_covs)

    def e_step(self, phi_j, mu_j, cov_j):
        w_ij = []
        rvs = []
        for i in range(self.K):
            rvs.append(multivariate_normal(mu_j[i], cov_j[i]))
        for i in range(self.N):
            rv_vals = [
                phi_j[i][rv_index] * rv.pdf(self.data[i])
                for rv_index, rv in enumerate(rvs)
            ]
            sum_rv_vals = sum(rv_vals)
            w = [rv_val / sum_rv_vals for rv_val in rv_vals]
            w_ij.append(w)
        return np.array(w_ij)

    def main(self):
        phi_j, mu_j, cov_j = self.initialize_params()
        print("Training...")
        for _ in tqdm(range(50)):
            w_ij = self.e_step(phi_j, mu_j, cov_j)
            phi_j, mu_j, cov_j = self.m_step(w_ij, mu_j)
        print("Training complete!!!")
        self.result = [phi_j, mu_j, cov_j]


if __name__ == "__main__":
    em = EM("EM.csv")
    em.main()
    em.viz_data(with_mean=True)
