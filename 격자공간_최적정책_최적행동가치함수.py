#최적상태가치함수 만들기
import numpy as np

class Environment2:
    def __init__(self):
        # 다른 객체 변수 필요없음!
        self.q_table = np.zeros((5, 5, 4)) #최적 행동 가치 함수

    def update(self):
        # update를 할 때마다 식 3.20에 따라 q_table의 값을 변경
        self.next_q_table = np.zeros((5, 5, 4))

        for row in range(5):
            for col in range(5):
                for action in range(4):
                    next_row, next_col, reward= self.get_next_state(row, col, action)
                    self.next_q_table[row, col, action] +=  reward + 0.9 * np.max(self.q_table[next_row, next_col][:])

        self.q_table = self.next_q_table.copy()

    def get_next_state(self, row, col, action):
        # update 함수에 사용할 state와 action을 넣으면 next state와 reward를 반환하는 함수
        if row == 0 and col == 1:
            next_row, next_col = 4, 1
            reward = 10

        elif row == 0 and col == 3:
            next_row, next_col = 2, 3
            reward = 5

        else:  # 일반적인 경우
            if action == 0:
                next_row, next_col = row - 1, col
            elif action == 1:
                next_row, next_col = row + 1, col
            elif action == 2:
                next_row, next_col = row, col - 1
            else: # elif action == 3:
                next_row, next_col = row, col + 1

            # 그리드 밖으로 나가는 경우
            if next_row < 0 or next_row >= 5 or next_col < 0 or next_col >= 5:
                reward = -1
                next_row, next_col = row, col
            else:
                reward = 0

        return next_row, next_col, reward
    
    def get_policy(self): # q_table 값에 따라 정책 화살표로 print
        # q_table 값에 따라 정책 화살표로 print
        dir_dict = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        direction_list = []
        for row in range(5):
            direction_list_row = []
            for col in range(5):
                if row == 0 and col == 1 or row == 0 and col == 3:
                    direction_list_row.append("* ")
                    continue
                action_value = ""
                
                # 최대값을 가지는 행동의 모든 인덱스를 가져옴
                max_idx_list = np.argwhere(self.q_table[row, col, :] == np.max(self.q_table[row, col, :])).flatten().tolist()

                for i in max_idx_list:
                    action_value += dir_dict[i]
                action_value += " " * (2-len(max_idx_list))

                direction_list_row.append(action_value)

            direction_list.append(direction_list_row)
        return direction_list


env = Environment2()
print("초기 상태 가치 함수")
print(np.max(env.q_table, axis=2))

for i in range(100):
    env.update()

print("\n최적 행동 가치 함수로 계산한 최적 상태 가치 함수  (100번 반복)")
print(np.max(env.q_table, axis=2))

print("\n최적 행동 가치 함수로 계산한 정책")
print(np.array(env.get_policy()))