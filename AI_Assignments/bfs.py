import sys


class PancakeSolver:
    def __init__(self, initial_seq):
        self.initial_sequence, self.method = initial_seq.strip().lower().split("-")
        self.initial_sequence = [
            self.initial_sequence[i * 2 : (i + 1) * 2]
            for i in range(len(self.initial_sequence) // 2)
        ]

    def flush(self):
        self.path = []
        self.heuristic_cost = 0
        self.done_sequences = []
        self.parent_tracker = {}
        self.solution_state = [f"{i+1}w" for i in range(len(self.initial_sequence))]

    def _printer(self, parent, spatula_pos=None, heuristic_dist=0):
        if not spatula_pos:
            joined_seq = "".join(parent)
        else:
            joined_seq = (
                "".join(parent[:spatula_pos]) + "|" + "".join(parent[spatula_pos:])
            )
        return "{} g:{}, h:{}".format(joined_seq, spatula_pos, heuristic_dist)

    def get_path(self):
        parents = [self.parent_tracker["".join(self.solution_state)]]
        while 1:
            try:
                parents.append(self.parent_tracker["".join(parents[-1]["parent"])])
            except KeyError:
                break
        parents = parents[::-1] + [{"parent": self.solution_state, "spatula_pos": None}]
        all_str = [self._printer(**x) for x in parents]
        return "\n".join(all_str)

    def heuristic_cost_calc(self):
        fn = lambda x: 0
        self.heuristic_cost += fn()

    def is_solution(self, sequence):
        return sequence == self.solution_state

    def _flip(self, pancake):
        if pancake[-1] == "w":
            return pancake[0] + "b"
        return pancake[0] + "w"

    def flipper(self, sequence, spatula_pos):
        seq_front, seq_after = sequence[:spatula_pos], sequence[spatula_pos:]
        seq_front = [self._flip(x) for x in seq_front]
        sequence = seq_front[::-1] + seq_after
        return sequence, spatula_pos

    def get_child_states(self, sequence):
        childrens = [self.flipper(sequence, i) for i in range(1, len(sequence) + 1)]
        childrens = [x for x in childrens if x[0] not in self.done_sequences]
        for c in childrens:
            try:
                _ = self.parent_tracker["".join(c[0])]
            except KeyError:
                self.parent_tracker["".join(c[0])] = {
                    "parent": sequence,
                    "spatula_pos": c[1],
                }
        return childrens

    def bfs(self, curr_sequence_list):
        i = 0
        while 1:
            cs, spatula_pos = curr_sequence_list.pop(0)
            if cs in self.done_sequences:
                continue
            if self.is_solution(cs):
                break
            self.path.append({"seq_lst": cs, "spatula_pos": spatula_pos})
            self.done_sequences.append(cs)
            curr_sequence_list += self.get_child_states(cs)
            i += 1
        return True

    def run(self):
        self.flush()
        if self.method == "b":
            self.bfs([[self.initial_sequence, 0]])
        return self.get_path()


if __name__ == "__main__":
    # 1w2b3b4w-b
    args = sys.argv
    c = PancakeSolver(args[1])
    print(c.run())
