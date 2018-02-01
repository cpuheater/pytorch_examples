import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Policy(nn.Module):
    def __init__(self, num_bandits, num_actions):
      super(Policy, self).__init__()
      self.linear = nn.Linear(num_bandits, num_actions, bias=False)
      self.linear.weight = torch.nn.Parameter(torch.ones(num_actions, num_bandits))

    def forward(self, input):
        one_hot = self.to_one_hot(input, 3)
        result = F.sigmoid(self.linear(Variable(one_hot)))
        return result

    def to_one_hot(self, action, depth):
      ones = torch.sparse.torch.eye(depth)
      return ones[action, :]

    def pull_arm(self, bandit, action):
      answer = np.random.randn(1)
      if bandit[action] > answer:
          return 1
      else:
          return -1



bandits = [[-5, -1, 0, 1], [-1, -5, 1, 0], [0, 1, -1, -5]]
num_bandits = len(bandits)
num_actions = len(bandits[0])

policy = Policy(num_bandits, num_actions)

optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)


rewards = np.zeros([num_bandits, num_actions])

for t in range(1000):

  state = np.random.randint(0, num_bandits)
  result = policy(state)
  if np.random.rand(1) < 0.25:
    action = np.random.randint(0, num_actions)
  else:
    action = np.argmax(result.data.numpy())
  reward = policy.pull_arm(bandits[state], action)
  loss = -torch.log(result[action]) * reward
  rewards[state, action] += reward
  policy.zero_grad()
  loss.backward()
  optimizer.step()
