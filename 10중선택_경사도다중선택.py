import numpy as np
import matplotlib.pyplot as plt

value = []
#표준정규분포를 따르는 10개의 슬롯머신 생성
for i in range(10):
    value.append(np.random.randn())
value = np.array(value)

rewards_mean=[[],[],[],[]]
for j in range(4):
    N = np.zeros(10)
    H = np.zeros(10) #선호도
    ㅠ = np.zeros(10) #소프트맥스
    ㅠ += 0.1
    rewards = []
    action_percent = []

    episode = 10000
    alpha = [0.01,0.4,0.2,0.6]

    for i in range(episode):
        action = np.random.choice(10, p=ㅠ)
        reward = np.random.randn() + value[action]
        rewards.append(reward)

        N[action] += 1
        H[action] = H[action] + alpha[j]*(reward - np.mean(rewards))*(1-ㅠ[action])
        H[:action] = H[:action] - alpha[j]*(reward - np.mean(rewards))*ㅠ[:action]
        H[action+1:] = H[action+1:] - alpha[j]*(reward - np.mean(rewards))*ㅠ[action+1:]
        ㅠ = np.exp(H)/np.sum(np.exp(H))
        action_percent.append(N[np.argmax(value)]/(i+1))
        rewards_mean[j].append(np.mean(rewards))
    
    action_percent = np.array(action_percent)
    plt.plot(action_percent*100, label='alpha = '+str(alpha[j]))
plt.legend()
plt.show()
plt.plot(rewards_mean[0], label='alpha = 0.01')
plt.plot(rewards_mean[1], label='alpha = 0.4')
plt.plot(rewards_mean[2], label='alpha = 0.2')
plt.plot(rewards_mean[3], label='alpha = 0.6')
plt.legend()
plt.show()
