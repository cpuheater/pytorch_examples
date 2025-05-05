import numpy as np
import torch


torch.manual_seed(1)
np.random.seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pull_arm(bandit):
      result = np.random.randn(1)
      if result > bandit:
          return 1
      else:
          return -1

bandits = [0.1, 1, -0.4, -5]
num_bandits = len(bandits)

w = torch.ones(num_bandits, requires_grad=True, dtype=torch.float)
rewards = np.zeros(num_bandits)

lr = 0.001

for t in range(1000):
  if np.random.rand(1) < 0.1:
    action = np.random.randint(0, num_bandits)
  else:
    action = np.argmax(w.data)
  reward = pull_arm(bandits[action])
  loss = -torch.log(w[action]) * reward
  rewards[action] += reward
  loss.backward()
  w.data -=  lr* w.grad.data
  w.grad.data.zero_()

print(rewards)