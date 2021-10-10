import math


class Node:
    def __init__(self, index=None, is_maximizer=True, value=None) -> None:
        self.childrens = []
        self.index = index
        self.is_maximizer = is_maximizer
        self.value = value


class AlphaBeta:
    def __init__(self, sequence) -> None:
        self.terminal_values = list(map(int, sequence.split(" ")))
        self.build_tree()
        self.visited = []

    def build_tree(self):
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
        if alpha >= beta:
            return alpha, beta
        self.visited.append(node.index)
        if node.childrens == []:
            return node.value, node.value
        for children in node.childrens:
            child_alpha, child_beta = self.alpha_beta_prune(children, alpha, beta)
            if node.is_maximizer:
                alpha = max(alpha, child_alpha, child_beta)
            else:
                beta = min(beta, child_alpha, child_beta)
        return alpha, beta

    def main(self):
        self.alpha_beta_prune(self.root_node, -math.inf, math.inf)
        self.visited = [x for x in self.visited if x is not None]
        self.pruned = [x for x in range(12) if x not in self.visited]
        print(" ".join(list(map(str, self.pruned))))


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
