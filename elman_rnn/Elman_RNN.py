import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import time
import torch.nn.init as init

dtype = torch.FloatTensor

input_size, hidden_size, output_size = 10, 9, 1
epochs = 300
num_time_steps = 10
lr = 0.1


time_steps = np.linspace(2, 5, num_time_steps)
data = np.sin(time_steps)
data.resize((num_time_steps, 1))

x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad=False)


lim = np.sqrt(6) / (np.sqrt(input_size + hidden_size))
w1 = np.random.uniform(-lim, lim, [input_size, hidden_size])
w1 = Variable(torch.from_numpy(w1).type(dtype), requires_grad=True)
lim = np.sqrt(6) / (np.sqrt(output_size + hidden_size))
w2 = np.random.uniform(-lim, lim, [hidden_size, output_size])
w2 = Variable(torch.from_numpy(w2).type(dtype), requires_grad=True)

def forward(x,w1, w2):
  hidden = torch.tanh(x.mm(w1))
  out = hidden.mm(w2)
  return  (out, hidden)

for i in range(epochs):
  total_loss = 0
  hidden_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=True)
  for j in range(x.size(0)):
    input = x[j:(j+1)]
    target = y[j:(j+1)]
    xh = torch.cat((input, hidden_state), 1)
    (pred, hidden_state) = forward(xh, w1, w2)
    loss = (pred - target).pow(2).sum()/2
    total_loss += loss
    loss.backward()
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    hidden_state = Variable(hidden_state.data)
  if i % 10 == 0:
     print("Epoch: {} loss {}".format(i, total_loss.data[0]))


hidden_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad=False)
predictions = []
for i in range(x.size(0)):
  input = x[i:(i+1)]
  xh = torch.cat((input, hidden_state), 1)
  (pred, hidden_state) = forward(xh, w1, w2)
  hidden_state = hidden_state
  predictions.append(pred.data.numpy().ravel()[0])


pl.scatter(time_steps[:-1], x.data.numpy(), s=90)
pl.scatter(time_steps[1:], predictions)
pl.show()
