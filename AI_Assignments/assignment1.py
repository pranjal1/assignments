"""
Author: Pranjal Dhakal
Assigment 1
Intro to Dear AI - Fall 2021
"""


class Node:
    """
    This class implements a node in the search tree.
    """

    def __init__(self, parent, sequence, parent_spatula_position):
        """
        parent: parent node of the current sequence
        sequence: current sequence obtained after flipping the parent sequence with the spatula
        parent_spatula_position: current sequence is obtained after flipping the pancakes in the parent in the parent_spatula_position position
        """
        self.parent = parent
        self.sequence = sequence
        self.parent_spatula_position = parent_spatula_position
        # hn is the heuristic cost of the current node
        self.hn = self.heuristic_cost()
        # parent_gn is the gn cost of the parent node
        # gn is the gn cost of the current node
        if parent is None:
            self.parent_gn = 0
            self.gn = 0
        else:
            self.parent_gn = self.parent.gn
            self.gn = self.parent_gn + self.parent_spatula_position
        # current node's hn is the sum of gn and hn
        self.fn = self.gn + self.hn
        # conflict_solve_value is used to handle the tie-breaker cases
        self.conflict_solve_value = self.conflict_solve()

    def __str__(self):
        # for printing and debug
        if self.parent is None:
            return "sequence {}, parent {}, parent spatula position {}".format(
                "".join(self.sequence), None, 0
            )
        return "sequence {}, parent {}, parent spatula position {}".format(
            "".join(self.sequence),
            "".join(self.parent.sequence),
            self.parent_spatula_position,
        )

    def __repr__(self):
        # for printing and debug
        if self.parent is None:
            return "sequence {}, parent {}, parent spatula position {}".format(
                "".join(self.sequence), None, 0
            )
        return "sequence {}, parent {}, parent spatula position {}".format(
            "".join(self.sequence),
            "".join(self.parent.sequence),
            self.parent_spatula_position,
        )

    def is_solution(self):
        # to check whether the current node is the solution
        return self.sequence == ["1w", "2w", "3w", "4w"]

    def heuristic_cost(self):
        # implementation of heuristic cost calculation
        # id of the pancakes out of their correct places is determined
        # pancake with the largest id that is out of place is the heuristic cost (hn)
        out_of_place = []
        for i, x in enumerate(self.sequence):
            pancake_num = int(x[0])
            if pancake_num - 1 != i:
                out_of_place.append(pancake_num)
        h_cost = 0
        if out_of_place:
            h_cost = max(out_of_place)
        return h_cost

    def conflict_solve(self):
        # implementation of the tie-breaker logic
        dct = {"w": "1", "b": "0"}
        fn = lambda x: int("".join([dct.get(x, x) for x in "".join(x)]))
        return fn(self.sequence)

    def flipper(self, sequence, spatula_pos):
        # takes the sequence of a node and the spatula position
        # obtains the resulting sequence by flipping the given sequence at spatula_pos
        _flip = lambda x: x[0] + "b" if x[-1] == "w" else x[0] + "w"
        seq_front, seq_after = sequence[:spatula_pos], sequence[spatula_pos:]
        seq_front = [_flip(x) for x in seq_front]
        sequence = seq_front[::-1] + seq_after
        return sequence, spatula_pos

    def get_child_nodes(self):
        # all the child nodes of the cuurent node is obtained
        # this is done by flipping the current node at all positions
        self.childrens = []
        for i in range(1, len(self.sequence) + 1):
            self.childrens.append(Node(self, *self.flipper(self.sequence, i)))


class NodeSimpler:
    """
    This class also implements a node in the search tree.
    The difference with the previous class is that it does not have any cost calculation associated with any node.
    The description of each class method is exactly same as the previous class.
    """

    def __init__(self, parent, sequence, parent_spatula_position):
        self.parent = parent
        self.sequence = sequence
        self.parent_spatula_position = parent_spatula_position

    def __str__(self):
        if self.parent is None:
            return "sequence {}, parent {}, parent spatula position {}".format(
                "".join(self.sequence), None, 0
            )
        return "sequence {}, parent {}, parent spatula position {}".format(
            "".join(self.sequence),
            "".join(self.parent.sequence),
            self.parent_spatula_position,
        )

    def __repr__(self):
        return "sequence {}, parent {}, parent spatula position {}".format(
            "".join(self.sequence),
            "".join(self.parent.sequence),
            self.parent_spatula_position,
        )

    def is_solution(self):
        return self.sequence == ["1w", "2w", "3w", "4w"]

    def flipper(self, sequence, spatula_pos):
        _flip = lambda x: x[0] + "b" if x[-1] == "w" else x[0] + "w"
        seq_front, seq_after = sequence[:spatula_pos], sequence[spatula_pos:]
        seq_front = [_flip(x) for x in seq_front]
        sequence = seq_front[::-1] + seq_after
        return sequence, spatula_pos

    def get_child_nodes(self):
        self.childrens = []
        for i in range(1, len(self.sequence) + 1):
            self.childrens.append(NodeSimpler(self, *self.flipper(self.sequence, i)))


class PancakeSolver:
    """
    This class solves the Burnt Pancake problem using either BFS or A* algorithm
    """

    def __init__(self, initial_seq):
        """
        initial_seq: initial sequence supplied by the user
        """
        # separating the sequence and the method to be used
        self.initial_sequence, self.method = initial_seq.strip().lower().split("-")
        self.initial_sequence = [
            self.initial_sequence[i * 2 : (i + 1) * 2]
            for i in range(len(self.initial_sequence) // 2)
        ]
        # initialize empty fringe
        self.fringe = []
        # keep track of the visited nodes
        self.visited_nodes = []
        # to keep track of the solution state, will be later used to backtrack back to user provided sequence
        self.solution_state = None

    def printer(self, parent, spatula_pos=None, hn=None, gn=None):
        # method to print the results
        if not spatula_pos:
            joined_seq = "".join(parent[:spatula_pos])
        else:
            joined_seq = (
                "".join(parent[:spatula_pos]) + "|" + "".join(parent[spatula_pos:])
            )
        if self.method == "b":
            return "{}".format(joined_seq)
        else:
            return "{} g:{}, h:{}".format(joined_seq, gn, hn)

    def a_star(self):
        # A* implementation

        # root node is the node from the user provided sequence.
        # it has None as parent and no parent spatula position
        root_node = Node(None, self.initial_sequence, 0)

        # append root node to the fringe for exploration
        self.fringe.append(root_node)

        while 1:
            # pop the first node in the fringe
            current_node = self.fringe.pop(0)
            # check if already visited (graph-search)
            if str(current_node) in self.visited_nodes:
                continue
            # check if current node is the solution
            if current_node.is_solution():
                # if successful assign the current node as solution state and exit while loop
                self.solution_state = current_node
                break
            # if not solution, add the visited node to visited node list
            self.visited_nodes.append(str(current_node))
            # get children nodes of the unsuccessful node and add it to the fringe
            current_node.get_child_nodes()
            self.fringe += current_node.childrens
            # sort the fringe using the node's fn value
            # if 2 or nodes have the same fn value, use the tie breaker value to resolve sorting
            self.fringe = sorted(
                self.fringe, key=lambda x: [x.fn, x.conflict_solve_value]
            )

    def bfs(self):
        # BFS implementation

        # root node is the node from the user provided sequence.
        # it has None as parent and no parent spatula position
        root_node = NodeSimpler(None, self.initial_sequence, 0)
        # append root node to the fringe for exploration
        self.fringe.append(root_node)

        while 1:
            # pop the first node in the fringe
            current_node = self.fringe.pop(0)
            # check if already visited (graph-search)
            if str(current_node) in self.visited_nodes:
                continue
            # check if current node is the solution
            if current_node.is_solution():
                # if successful assign the current node as solution state and exit while loop
                self.solution_state = current_node
                break
            # if not solution, add the visited node to visited node list
            self.visited_nodes.append(str(current_node))
            # get children nodes of the unsuccessful node and add it to the fringe
            current_node.get_child_nodes()
            self.fringe += current_node.childrens

    def run(self):
        if self.method == "a":
            self.a_star()
            solution_path = []
            c_state = self.solution_state
            while 1:
                solution_path.append(c_state)
                c_state = c_state.parent
                if c_state is None:
                    break
            solution_path = solution_path[::-1]
            solution_txt = []
            for cs, ns in zip(solution_path[:-1], solution_path[1:]):
                solution_txt.append(
                    self.printer(
                        cs.sequence, ns.parent_spatula_position, hn=cs.hn, gn=cs.gn
                    )
                )
            solution_txt.append(
                self.printer(
                    self.solution_state.sequence,
                    None,
                    hn=self.solution_state.hn,
                    gn=self.solution_state.gn,
                )
            )
            return "\n".join(solution_txt)
        else:
            self.bfs()
            solution_path = []
            c_state = self.solution_state
            while 1:
                solution_path.append(c_state)
                c_state = c_state.parent
                if c_state is None:
                    break
            solution_path = solution_path[::-1]
            solution_txt = []
            for cs, ns in zip(solution_path[:-1], solution_path[1:]):
                solution_txt.append(
                    self.printer(cs.sequence, ns.parent_spatula_position)
                )
            solution_txt.append(self.printer(self.solution_state.sequence, None))
            return "\n".join(solution_txt)


if __name__ == "__main__":
    pattern = input("Enter initial sequence\n")
    solver = PancakeSolver(pattern)
    print(solver.run())
