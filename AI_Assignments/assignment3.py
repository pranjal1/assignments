import random
from tqdm import tqdm

random.seed(1)


class Cell:
    def __init__(self, index, title, reward):
        self.index = index
        if title == "wall":
            self.actions = []
            self.reward = reward
        elif title == "goal":
            self.actions = ["exit"]
            self.reward = reward
        elif title == "forbidden":
            self.actions = ["exit"]
            self.reward = reward
        else:
            # setting in this way will automatically ensure tie-breaker
            self.actions = ["up", "right", "down", "left"]
            self.reward = reward
        self.q_values = {a: 0 for a in self.actions}


class QLearning:
    def __init__(self, sequence):
        self.sequence = sequence
        self.alpha = 0.3
        self.discount_factor = 0.1
        self.living_reward = -0.1
        self.goal_reward = +100
        self.forbidden_reward = -100
        self.epsilon = 0.5
        self.form_grid()

    def parse(self):
        splitted = self.sequence.split(" ")
        self.int_splitted = list(map(int, splitted[:4]))
        self.goal_index_1 = self.int_splitted[0]
        self.goal_index_2 = self.int_splitted[1]
        self.forbidden_index = self.int_splitted[2]
        self.wall_index = self.int_splitted[3]
        if len(splitted) == 5:
            self.mode = "policy"
        else:
            self.mode = "q_print"
            self.required_cell = int(splitted[-1])

    def get_successor(self, cell):
        if random.random() < self.epsilon:
            action = max(list(cell.q_values.items()), key=lambda x: x[1])[0]
        else:
            action = random.sample(cell.actions, 1)[0]

        if action == "up":
            succesor_index = cell.index + 4
        elif action == "down":
            succesor_index = cell.index - 4
        elif action == "right":
            if cell.index % 4 == 0:
                succesor_index = cell.index
            else:
                succesor_index = cell.index + 1
        elif action == "left":
            if (cell.index - 1) % 4 == 0:
                succesor_index = cell.index
            else:
                succesor_index = cell.index - 1
        else:
            succesor_index = cell.index

        if (
            succesor_index > 16
            or succesor_index < 1
            or succesor_index == self.wall_index
        ):
            succesor_index = cell.index
        return action, succesor_index

    def form_grid(self):
        self.parse()
        self.cells = {
            i: Cell(i, "normal", self.living_reward)
            for i in range(1, 17)
            if i not in self.int_splitted
        }
        self.cells[self.goal_index_1] = Cell(
            self.goal_index_1, "goal", self.goal_reward
        )
        self.cells[self.goal_index_2] = Cell(
            self.goal_index_2, "goal", self.goal_reward
        )
        self.cells[self.forbidden_index] = Cell(
            self.forbidden_index, "forbidden", self.forbidden_reward
        )
        self.cells[self.wall_index] = Cell(self.wall_index, "wall", 0)

    def final_policy(self):
        policy = []
        for index, cell in self.cells.items():
            if index in [self.goal_index_1, self.goal_index_2]:
                policy.append([index, "goal"])
            elif index == self.forbidden_index:
                policy.append([index, "forbid"])
            elif index == self.wall_index:
                policy.append([index, "wall-square"])
            else:
                policy.append(
                    [index, max(cell.q_values.items(), key=lambda x: x[1])[0]]
                )
        policy = sorted(policy, key=lambda x: x[0])
        for ind, pol in policy:
            print(f"{ind}  {pol}")

    def main(self):
        for i in tqdm(range(10000)):
            current_cell_index = random.sample(
                [i for i in range(1, 17) if i not in self.int_splitted], 1
            )[0]
            while current_cell_index not in [
                self.goal_index_1,
                self.goal_index_2,
                self.forbidden_index,
            ]:
                current_cell = self.cells[current_cell_index]
                action, succesor_index = self.get_successor(current_cell)
                succesor_cell = self.cells[succesor_index]
                reward = succesor_cell.reward
                old_q_value = current_cell.q_values[action]
                temporal_diff = (
                    reward
                    + self.discount_factor * max(succesor_cell.q_values.values())
                    - old_q_value
                )
                current_cell.q_values[action] = current_cell.q_values[
                    action
                ] + self.alpha * (temporal_diff)
                self.cells[current_cell_index].q_values = current_cell.q_values
                current_cell_index = succesor_index
        self.final_policy()


# if __name__ == "__main__":
#     # ql = QLearning("15 12 8 6 q 11")
#     # ql = QLearning("15 12 8 6 p")
#     ql = QLearning("10 8 9 6 p")
#     ql.main()

if __name__ == "__main__":
    pattern = input("Enter sequence\n")
    QLearning(pattern).main()
