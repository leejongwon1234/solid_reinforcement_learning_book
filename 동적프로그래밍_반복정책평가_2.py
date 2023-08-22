import numpy as np

# 4x4 격자 세계에서의 정책 평가
# 상태 가치 함수를 반복 정책 평가로 구하는 코드
# 상태 가치 함수를 구하는 것이 목적이므로 정책은 무작위로 설정

class Environment:
    def __init__(self):
        self.v = np.zeros((4, 4))

    def update(self):
        # 새로운 v와 기존의 v가 거의 같아질 때까지 반복
        while True:
            new_v = np.zeros((4, 4))
            for row in range(4):
                for col in range(4):
                    if row == 0 and col == 0 or row == 3 and col == 3:
                        continue
                    new_v[row, col] = self.get_new_v(row, col)

            if np.sum(np.abs(self.v - new_v)) < 1e-4:
                break

            self.v = new_v

    def judge_fail(self, row, col):
        action_table = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        fail = 0
        value_action = []
        for action in range(4):
            row_tar = row + action_table[action][0]
            col_tar = col + action_table[action][1]

            if row_tar < 0 or row_tar > 3 or col_tar < 0 or col_tar > 3:
                fail += 1
            else:
                value_action.append(action)

        return fail, value_action

    def get_new_v(self, row, col):
        # update 함수를 위한 각 상태에 대한 다음 가치 함수의 값을 가져옴
        action_table = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        new_v = 0
        fail, value_action = self.judge_fail(row,col)
        p = 1/(4-fail)

        for action in value_action:
            row_tar = row + action_table[action][0]
            col_tar = col + action_table[action][1]

            new_v += (p * (-1 + self.v[row_tar, col_tar]))  

        return new_v

    def get_policy(self):
        action_table = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        dir_table = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        direction_list = []
        for row in range(4):
            direction_list_row = []
            for col in range(4):
                if row == 0 and col == 0 or row == 3 and col == 3:
                    direction_list_row.append("* ")
                    continue
                direction = ""
                action_value = []
                for action in range(4):
                    row_tar = row + action_table[action][0]
                    col_tar = col + action_table[action][1]

                    if row_tar < 0 or row_tar > 3 or col_tar < 0 or col_tar > 3:
                        action_value.append(-999)
                    else:
                        action_value.append(self.v[row_tar, col_tar])

                max_idx_list = np.argwhere(np.isclose(action_value, np.max(action_value))).flatten().tolist()

                for i in max_idx_list:
                    direction += dir_table[i]
                direction += " " * (2-len(max_idx_list))
                direction_list_row.append(direction)
            direction_list.append(direction_list_row)
        return np.array(direction_list)


env = Environment()
print("초기 가치 함수")
print(env.v)

env.update()
print("\n무작위 정책으로 수렴할 때까지 반복한 가치 함수")
print(env.v)

print("\n가치 함수로부터 행동을 결정하는 정책")
print(env.get_policy())