"""
Author: Pranjal Dhakal
AI assignment 2
"""

import math


class Node:
    """
    Represents a node in the Tree.
    """

    def __init__(
        self, index: int = None, is_maximizer: bool = True, value: int = None
    ) -> None:
        """
        Args:
            index: Index of the terminal node.
            is_maximizer: whether the node is in maximizer or minimizer mode
            value: the value for terminal node.
        """
        self.childrens = []
        self.index = index
        self.is_maximizer = is_maximizer
        self.value = value


class AlphaBeta:
    """
    This class solves the AlphaBeta pruning.
    """

    def __init__(self, sequence: str) -> None:
        """
        Args:
            sequence: User provided sequence of the terminal nodes.
        """
        # split the sequence into individual integers
        self.terminal_values = list(map(int, sequence.split(" ")))
        # builds the tree according to the figure given in the assignment
        self.build_tree()
        # This keeps track of the nodes that are visited. This is used to later get the pruned terminal nodes.
        self.visited = []

    def build_tree(self):
        """"
        This method build the tree as per the figure in the assignment.
        """
        self.root_node = Node(is_maximizer=True)
        for i in range(3):
            second_layer_node = Node(is_maximizer=False)
            for j in range(2):
                third_layer_node = Node(is_maximizer=True)
                for k in range(2):
                    third_layer_node.childrens.append(
                        Node(
                            index=i * 4 + j * 2 + k,
                            is_maximizer=False,
                            value=self.terminal_values[i * 4 + j * 2 + k],
                        )
                    )
                second_layer_node.childrens.append(third_layer_node)
            self.root_node.childrens.append(second_layer_node)

    def alpha_beta_prune(self, node, alpha, beta):
        # if alpha > beta, dont explore its childrens
        if alpha >= beta:
            return alpha, beta
        # note the node that is visited
        self.visited.append(node.index)
        if node.childrens == []:
            return node.value, node.value
        for children in node.childrens:
            # recursion for exploring the tree and updating the alpha and beta values
            child_alpha, child_beta = self.alpha_beta_prune(children, alpha, beta)
            if node.is_maximizer:
                alpha = max(alpha, child_alpha, child_beta)
            else:
                beta = min(beta, child_alpha, child_beta)
        return alpha, beta

    def main(self):
        # initially alpha and beta are -inf and inf.
        alpha, beta = self.alpha_beta_prune(self.root_node, -math.inf, math.inf)
        self.visited = [x for x in self.visited if x is not None]
        # pruned nodes are those that are not visited
        self.pruned = [x for x in range(12) if x not in self.visited]
        print(" ".join(list(map(str, self.pruned))))


# if __name__ == "__main__":
#     pattern = input("Enter initial sequence\n")
#     AlphaBeta(pattern).main()

if __name__ == "__main__":
    patterns = [
        "2 4 13 11 1 3 3 7 3 3 2 2",
        "1 4 2 6 8 7 3 7 2 3 2 2",
        "15 4 12 16 10 7 3 1 2 3 2 2",
        "1 4 12 16 1 7 3 1 2 8 2 2",
        "1 4 12 16 1 7 3 1 2 8 10 2",
    ]
    for p in patterns:
        print(p)
        AlphaBeta(p).main()

