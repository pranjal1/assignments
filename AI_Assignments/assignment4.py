"""
Author: Pranjal Dhakal
Assigment 4
Intro to Dear AI - Fall 2021
"""

import re
import math


class BinaryClassifier:
    def _init_(self, user_input) -> None:
        self.user_input = user_input
        self.w = [0.0, 0.0]

    def parse_input(self):
        """
        Parses the user input
        """
        self.task = self.user_input[0]
        ip_pattern = r"\((?P<f1>[+-]?\d+).+?(?P<f2>[+-]?\d+).+?(?P<y>[+-]\d+)\)"
        self.data = [
            list(map(float, (m["f1"], m["f2"], m["y"])))
            for m in re.finditer(ip_pattern, self.user_input)
        ]

    def perceptron(self):
        """
        Implementing perceptron
        """
        for _ in range(100):
            for d1, d2, y in self.data:
                prediction = 1 if self.w[0] * d1 + self.w[1] * d2 >= 0 else -1
                if prediction != y:
                    self.w[0] += y * d1
                    self.w[1] += y * d2
        return " ".join(map(str, self.w))

    def sigmoid(self, z):
        """
        Sigmoid function
        """
        return 1 / (1 + math.exp(-z))

    def logistic_regression(self):
        """
        Implementing Logistic regression
        """
        changed_ys_data = [(d1, d2, 0 if y == -1 else 1) for d1, d2, y in self.data]
        for _ in range(100):
            for d1, d2, y in changed_ys_data:
                gz = self.sigmoid(self.w[0] * d1 + self.w[1] * d2)
                diff = y - gz
                self.w[0] += 0.1 * (diff * d1)
                self.w[1] += 0.1 * (diff * d2)
        final_pred = [
            round(self.sigmoid(self.w[0] * d1 + self.w[1] * d2), 2)
            for d1, d2, _ in self.data
        ]
        return " ".join(map(str, final_pred))

    def main(self):
        self.parse_input()
        if self.task == "P":
            return self.perceptron()
        return self.logistic_regression()


if __name__ == "__main__":
    ip = input("Input?\n")
    b = BinaryClassifier(ip)
    print(b.main())