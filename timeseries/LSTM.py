import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

training_set = pd.read_csv('./monthly-lake-erie-levels-1921-19.csv')
training_set = training_set.iloc[:,1:2].values


plt.plot(training_set, label = 'Monthly lake erie levels')
plt.show()


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


seq_length = 7
learning_rate = 0.01
input_size = 1
hidden_size = 5
num_classes = 1
num_layers = 1
num_epochs = 400

x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.7)
test_size = len(y) - train_size
trainX = torch.tensor(np.array(x[0:train_size]), requires_grad=True, dtype=torch.float)
testX = torch.tensor(np.array(x[train_size:len(x)]), requires_grad=True, dtype=torch.float)
trainY = torch.tensor(np.array(y[0:train_size]), requires_grad=True, dtype=torch.float)
testY = torch.tensor(np.array(y[train_size:len(y)]), requires_grad=True, dtype=torch.float)


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros((
            self.num_layers, x.size(0), self.hidden_size), requires_grad=True)
        c_0 = torch.zeros((
            self.num_layers, x.size(0), self.hidden_size), requires_grad=True)
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


lstm.eval()
test_predict = lstm(testX)

test_predict = test_predict.data.numpy()
testY = testY.data.numpy()
plt.plot(testY)
plt.plot(test_predict)
plt.show()