import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pylab as pl
import torch.nn.init as init
import math

dtype = torch.FloatTensor

num_time_steps = 10
input_size = 1
hidden_size = 16
output_size = 1
lr=0.01

time_steps = np.linspace(0, 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.Tensor(data[:-1]).type(dtype).view(1, num_time_steps - 1, 1)
y = torch.Tensor(data[1:]).type(dtype).view(1, num_time_steps - 1, 1)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        for p in self.rnn.parameters():
          init.normal(p)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
       out, hidden_prev = self.rnn(x, hidden_prev)
       out = out.view(-1, hidden_size)
       out = torch.stack(self.linear(out), dim=1)
       return out, hidden_prev

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)

hidden_prev = Variable(torch.zeros(1, 1, hidden_size))
x, y = Variable(x), Variable(y)

for iter in range(1000):
      output, hidden_prev = model(x, hidden_prev)
      hidden_prev = Variable(hidden_prev.data)
      loss = criterion(output, y)
      model.zero_grad()
      loss.backward()
      optimizer.step()
      if iter % 100 == 0:
        print("Iteration: {} loss {}".format(iter, loss.data[0]))

predictions = []
input = x[:, 0,:]
for _ in range(x.shape[1]):
  input = input.view(1, 1, 1)
  (pred, hidden_prev) = model(input, hidden_prev)
  input = pred
  hidden_prev = hidden_prev
  predictions.append(pred.data.numpy().ravel()[0])

x = x.data.numpy().ravel()
y = y.data.numpy()
pl.scatter(time_steps[:-1], x.ravel(), s=90)
pl.plot(time_steps[:-1], x.ravel())
pl.scatter(time_steps[1:], predictions)
pl.show()







