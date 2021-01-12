
import torch
import numpy as np
import torch.nn as nn
import pylab as pl

torch.manual_seed(1)

dtype = torch.float32
input_size, hidden_size, output_size = 7, 6, 1
epochs = 200
seq_length = 20
lr = 0.1

data_time_steps = np.linspace(2, 10, seq_length + 1)
data = np.sin(data_time_steps)
data.resize((seq_length + 1, 1))

x = torch.tensor(data[:-1], dtype=dtype, requires_grad=False)
y = torch.tensor(data[1:], dtype=dtype, requires_grad=False)

w1 = torch.normal(0.0, 0.4, size=(input_size, hidden_size), dtype=dtype, requires_grad=True)
w2 = torch.normal(0.0, 0.3, size=(hidden_size, output_size), dtype=dtype, requires_grad=True)


def forward(input, context_state, w1, w2):
  xh = torch.cat((input, context_state), 1)
  context_state = torch.tanh(xh.mm(w1))
  out = context_state.mm(w2)
  return  (out, context_state)

criterion = nn.MSELoss()
for i in range(epochs):
  total_loss = 0
  context_state = torch.zeros((1, hidden_size), dtype=dtype)
  for j in range(x.size(0)):
    input = x[j:(j+1)]
    target = y[j:(j+1)]
    (pred, context_state) = forward(input, context_state, w1, w2)
    loss = criterion(pred, target)
    total_loss += loss
    loss.backward(retain_graph=True)
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    context_state = context_state.clone()
  if i % 10 == 0:
     print("Epoch: {} loss {}".format(i, total_loss.item()))


context_state = torch.zeros((1, hidden_size), dtype=dtype)
predictions = []

for i in range(x.size(0)):
  input = x[i:i+1]
  (pred, context_state) = forward(input, context_state, w1, w2)
  context_state = context_state
  predictions.append(pred.data.numpy().ravel()[0])


pl.scatter(data_time_steps[:-1], x.data.numpy(), s=90, label="Actual")
pl.scatter(data_time_steps[1:], predictions, label="Predicted")
pl.legend()
pl.show()
