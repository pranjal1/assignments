"""
Author: Pranjal Dhakal
Q-Learning AI assignment 3
"""


import random

random.seed(1)


class Cell:
    """
    This class represents each of the 16 cells.
    Cells can either be wall, goal, forbidden or normal.
    Depending on the type, the actions that are permitted for these cell type are different.
    """

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
        # initial q_values for each cell is 0
        self.q_values = {a: 0.0 for a in self.actions}


class QLearning:
    """
    This class performs the Q-Learning
    """

    def __init__(self, sequence):
        # Get the initial sequence from the user and set Q-Learning parameters
        self.sequence = sequence
        self.alpha = 0.3
        self.discount_factor = 0.1
        self.living_reward = -0.1
        self.goal_reward = +100
        self.forbidden_reward = -100
        self.epsilon = 0.5
        # Form the 16 cells grid environment
        self.form_grid()

    def parse(self):
        # This method parses the user sequence
        # and gets index for goals, wall and forbidden cells.
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
        # This method calculates the successor cell
        # based on the current cell, the actions that can be performed from the cell

        if random.random() < self.epsilon:
            # select action that has the highest q-value
            action = max(list(cell.q_values.items()), key=lambda x: x[1])[0]
        else:
            # randomly select action based on the epsilon value
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
            # if the next position is invalid stay in the same index
            succesor_index = cell.index
        return action, succesor_index

    def form_grid(self):
        # This method forms the environment
        # It forms normal, goal, forbidden and wall cells based on the sequence supplied by user.
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
        # printing the policy for each cell
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
        return "\n".join([f"{ind}    {pol}" for ind, pol in policy])

    def get_result(self):
        # return result based on the whether the user sequence has p or q
        if self.mode == "policy":
            return self.final_policy()
        else:
            required_cell = self.cells[self.required_cell]
            required_cell.q_values = {
                k: round(v, 2) for k, v in required_cell.q_values.items()
            }
            return_str = ""
            return_str += "up    {:.1f}\n".format(required_cell.q_values["up"])
            return_str += "right    {:.1f}\n".format(required_cell.q_values["right"])
            return_str += "down    {:.1f}\n".format(required_cell.q_values["down"])
            return_str += "left    {:.1f}".format(required_cell.q_values["left"])
            return return_str

    def main(self):
        # main loop
        for i in range(100000):
            # select first cell randomly from among normal cells
            # current_cell_index = random.sample(
            #     [i for i in range(1, 17) if i not in self.int_splitted], 1
            # )[0]
            current_cell_index = 2
            # loop until goal or forbidden state is reached
            while current_cell_index not in [
                self.goal_index_1,
                self.goal_index_2,
                self.forbidden_index,
            ]:
                # uses the Q-Learning Bellman equation to update q-values for each cells.
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
        print(self.get_result())


# if __name__ == "__main__":
#     # ql = QLearning("15 12 8 6 q 11")
#     # ql = QLearning("15 12 8 6 p")
#     ql = QLearning("10 8 9 6 p")
#     ql.main()

if __name__ == "__main__":
    pattern = input("Enter sequence\n")
    QLearning(pattern).main()
