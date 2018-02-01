import numpy as np
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

def pull_arm(bandit):
      result = np.random.randn(1)
      if result > bandit:
          return 1
      else:
          return -1

bandits = [0.1, 1, -0.4, -5]
num_bandits = len(bandits)

w = Variable(torch.ones(num_bandits).type(dtype), requires_grad=True)
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