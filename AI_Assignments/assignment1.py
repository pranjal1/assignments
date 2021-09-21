class Node:
    def __init__(self, parent, sequence, parent_spatula_position):
        self.parent = parent
        self.sequence = sequence
        self.parent_spatula_position = parent_spatula_position
        self.hn = self.heuristic_cost()
        if parent is None:
            self.parent_gn = 0
            self.gn = 0
        else:
            self.parent_gn = self.parent.gn
            self.gn = self.parent_gn + self.parent_spatula_position
        self.fn = self.gn + self.hn
        self.conflict_solve_value = self.conflict_solve()

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
        return self.sequence == ["1w", "2w", "3w", "4w"]

    def heuristic_cost(self):
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
        dct = {"w": "1", "b": "0"}
        fn = lambda x: int("".join([dct.get(x, x) for x in "".join(x)]))
        return fn(self.sequence)

    def flipper(self, sequence, spatula_pos):
        _flip = lambda x: x[0] + "b" if x[-1] == "w" else x[0] + "w"
        seq_front, seq_after = sequence[:spatula_pos], sequence[spatula_pos:]
        seq_front = [_flip(x) for x in seq_front]
        sequence = seq_front[::-1] + seq_after
        return sequence, spatula_pos

    def get_child_nodes(self):
        self.childrens = []
        for i in range(1, len(self.sequence) + 1):
            self.childrens.append(Node(self, *self.flipper(self.sequence, i)))


class NodeSimpler:
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
    def __init__(self, initial_seq):
        self.initial_sequence, self.method = initial_seq.strip().lower().split("-")
        self.initial_sequence = [
            self.initial_sequence[i * 2 : (i + 1) * 2]
            for i in range(len(self.initial_sequence) // 2)
        ]
        self.fringe = []
        self.visited_nodes = []
        self.solution_state = None

    def printer(self, parent, spatula_pos=None, hn=None, gn=None):
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
        root_node = Node(None, self.initial_sequence, 0)
        self.fringe.append(root_node)

        while 1:
            current_node = self.fringe.pop(0)
            if str(current_node) in self.visited_nodes:
                continue
            if current_node.is_solution():
                self.solution_state = current_node
                break
            self.visited_nodes.append(str(current_node))
            current_node.get_child_nodes()
            self.fringe += current_node.childrens
            self.fringe = sorted(
                self.fringe, key=lambda x: [x.fn, x.conflict_solve_value]
            )

    def bfs(self):
        root_node = NodeSimpler(None, self.initial_sequence, 0)
        self.fringe.append(root_node)

        while 1:
            current_node = self.fringe.pop(0)
            if str(current_node) in self.visited_nodes:
                continue
            if current_node.is_solution():
                self.solution_state = current_node
                break
            self.visited_nodes.append(str(current_node))
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
    test_cases = [
        "1b2b3w4b-a",
        "1w2b3w4b-a",
        "1w2b3w4b-b",
        "1w2b3b4w-b",
    ]
    for t in test_cases:
        solver = PancakeSolver(t)
        print(t)
        print("-" * 20)
        print(solver.run())
        print("*" * 50)
