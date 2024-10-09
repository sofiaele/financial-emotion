import torch.nn as nn
import torch

# Define the MLP model
class MLPRegressor(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        #self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.relu(self.fc(x))
        x = self.fc(x)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x