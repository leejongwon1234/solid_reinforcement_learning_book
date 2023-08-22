import numpy as np
import matplotlib.pyplot as plt

# 가치 반복 : 도박사의 문제
# 동전 앞면이 나올 확률 0.25
# 동전 뒷면이 나올 확률 0.75
# 동전 앞면이 나올 경우 보상 1
# 동전 뒷면이 나올 경우 보상 0

# 할인율 1
# 상태 1 ~ 99
class Environment3:
    prob_front = 0.25
    prob_back = 1 - prob_front

    def __init__(self):
        self.v = np.zeros(100)
        
    def update(self):
        while True:
            # 상태 가치 함수를 업데이트
            new_v = np.zeros(100)
            for state in range(1, 100):
                action_value = []
                for action in range(1, min(state, 100 - state) + 1):
                    now_action_value = 0
                    # 동전 앞면이 나올 경우
                    if state + action == 100:
                        now_action_value = self.prob_front * (1)
                    else:
                        now_action_value = self.prob_front * (0 + self.v[state + action])
                    
                    # 동전 뒷면이 나올 경우
                    if state - action == 0:
                        now_action_value += 0
                    else:
                        now_action_value += self.prob_back * (0 + self.v[state - action])

                    action_value.append(now_action_value)
                new_v[state] = np.max(action_value)
            if np.sum(np.abs(self.v - new_v)) < 1e-4:
                break
            self.v = new_v

    def get_policy(self):
        policy = np.zeros(100)
        for state in range(1, 100):
            action_value = []
            for action in range(1, min(state, 100 - state) + 1):
                now_action_value = 0
                
                # 동전 앞면이 나올 경우
                if state + action == 100:
                    now_action_value = self.prob_front * (1)
                else:
                    now_action_value = self.prob_front * (0 + self.v[state + action])
                # 동전 뒷면이 나올 경우
                if state - action == 0:
                    now_action_value += 0
                else:
                    now_action_value += self.prob_back * (0 + self.v[state - action])

                action_value.append(now_action_value)
            policy[state] = np.argmax(action_value) + 1
        return policy
    
    def plot_V(self):
        plt.ylim(0, 1)
        plt.title("Value Function")
        plt.plot(range(1, 100), self.v[1:])
        plt.show()

    def plot_policy(self):
        policy = self.get_policy()
        plt.title("Policy")
        plt.ylim(0, 50)
        plt.bar(range(1, 100), policy[1:])
        plt.show()


env = Environment3()
env.update()

env.plot_V()
env.plot_policy()