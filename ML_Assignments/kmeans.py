"""
Author: Pranjal Dhakal
K-means algorithm
"""


import csv
import numpy as np
import matplotlib.pyplot as plt

import random


class KMeans:
    """
    Class for k-means clustering
    """
    def __init__(self, data_path:str, k:int=2):
        """
        data_path: Path of the csv file with x, y value in each line
        k: number of clusters
        """
        data = []
        with open(data_path) as csv_file:
            csv_reader = csv.reader(
                csv_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC
            )
            for row in csv_reader:
                data.append(row)
        self.data = np.array(data)
        self.k = k

    def cluster(self):
        # randomly initialize the k cluster means from the dataset
        cluster_random_init = random.sample(list(range(self.data.shape[0])), 2)
        cluster_means = self.data[cluster_random_init, :]
        # Loop until convergence
        while True:
            cluster_means_expanded = [
                np.tile(x, (self.data.shape[0], 1)) for x in cluster_means
            ]
            # calculate distance of each point from the cluster mean
            dist = [
                np.sqrt(np.sum(np.square(self.data - x), axis=1))
                for x in cluster_means_expanded
            ]
            dist = np.array(dist)
            # assign each point to one of k clusters based on
            # which of the k cluster means is the closest to the point.
            assignment = np.argmin(dist, axis=0)
            new_clusters_data = [
                self.data[assignment == x] for x in np.unique(assignment)
            ]
            # Calculate the new cluster mean
            new_cluster_means = np.array(
                [np.mean(x, axis=0) for x in new_clusters_data]
            )
            # break the loop if the cluster mean do not change
            if np.isclose(np.sum(cluster_means - new_cluster_means), 0):
                break
            cluster_means = new_cluster_means
        self.cluster_means = cluster_means
        self.assignment = assignment
        self.clusters = new_clusters_data
        self.WC = np.sum(np.min(dist, axis=0))
        print(self.WC)

    def plot(self):
        # plot the clusters
        plt.figure(figsize=(10, 10))
        for i, c in enumerate(self.clusters):
            plt.scatter(c[:, 0], c[:, 1])
        plt.scatter(
            self.cluster_means[0, 0], self.cluster_means[0, 1], c="black", s=100
        )
        plt.text(
            self.cluster_means[0, 0],
            self.cluster_means[0, 1] - 0.5,
            "c1",
            size=30,
            color="black",
        )
        plt.scatter(
            self.cluster_means[1, 0], self.cluster_means[1, 1], c="green", s=100
        )
        plt.text(
            self.cluster_means[1, 0],
            self.cluster_means[1, 1] - 0.5,
            "c2",
            size=30,
            color="green",
        )
        plt.show()

if __name__ == "__main__":
    km = KMeans("data1.csv")
    km.cluster()
    km.plot()
    print("Cluter means --> {}".format(km.cluster_means))

    km = KMeans("data2.csv")
    km.cluster()
    km.plot()
    print("Cluter means --> {}".format(km.cluster_means))