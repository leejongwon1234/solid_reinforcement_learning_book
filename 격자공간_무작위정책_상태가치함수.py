# 격자공간 5x5
# action 4개 : 상하좌우
# 상태 25개 : (0,0) ~ (4,4)
# (0,1)상태에서는 모든 행동에 대해 보상 10
# (0,1)상태에서 (4,1)상태로 이동
# (0,3)상태에서는 모든 행동에 대해 보상 5
# (0,3)상태에서 (2,3)상태로 이동
# 격자 공간을 벗어나는 행동은 보상 -1
# 나머지 상태에서는 보상 0
# 감가율 0.9

import numpy as np

class Environment:
    def __init__(self):
        self.value_table = np.zeros((5, 5))
        self.next_value_table = self.value_table.copy()

    def step(self, state:tuple[int, int], action)->tuple[int, int, int]: # (next_row, next_col), reward

        if state[0] == 0 and state[1] == 1:
            next_row, next_col = 4, 1
            reward = 10

        elif state[0] == 0 and state[1] == 3:
            next_row, next_col = 2, 3
            reward = 5

        else:  # 일반적인 경우
            if action == 0:
                next_row, next_col = state[0] - 1, state[1]
            elif action == 1:
                next_row, next_col = state[0] + 1, state[1]
            elif action == 2:
                next_row, next_col = state[0], state[1] - 1
            else: # elif action == 3:
                next_row, next_col = state[0], state[1] + 1

            # 그리드 밖으로 나가는 경우
            if next_row < 0 or next_row >= 5 or next_col < 0 or next_col >= 5:
                reward = -1
                next_row, next_col = state[0], state[1]
            else:
                reward = 0

        return next_row, next_col, reward
        
    def update(self):
        self.next_value_table = self.value_table.copy()

        for row in range(5):
            for col in range(5):
                self.next_value_table[row, col] = 0
                for action in range(4):
                    next_row,next_col, reward= self.step((row, col), action)
                    self.next_value_table[row, col] +=  0.25 * (reward + 0.9 * self.value_table[next_row, next_col])
        self.value_table = self.next_value_table.copy()

    def get_poilcy(self):
        dir_table = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        direction_list = []
        for row in range(5):
            direction_list_row=[]
            for col in range(5):
                if row == 0 and col == 1 or row == 0 and col == 3:
                    direction_list_row.append("*")
                    continue
                action_value = []
                for action in range(4):
                    if action == 0:
                        try:
                            action_value.append(self.value_table[row - 1, col])
                        except:
                            action_value.append(-999)
                    elif action == 1:
                        try:
                            action_value.append(self.value_table[row + 1, col])
                        except:
                            action_value.append(-999)
                    elif action == 2:
                        try:
                            action_value.append(self.value_table[row, col - 1])
                        except:
                            action_value.append(-999)
                    else:
                        try:
                            action_value.append(self.value_table[row, col + 1])
                        except:
                            action_value.append(-999)

                direction_list_row.append(dir_table[np.argmax(action_value)])
            direction_list.append(direction_list_row)
        return direction_list




env = Environment()
print("초기 상태 가치 함수")
print(env.value_table)

for i in range(100):
    env.update()

print("\n무작위 행동으로 수렴된 상태 가치 함수 (100번 반복)")
print(env.value_table)

print("\n상태 가치 함수로 계산한 정책")
print(np.array(env.get_poilcy()))