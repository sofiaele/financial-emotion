import torch.nn as nn


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        #self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.relu(self.fc(x))
        x = self.fc(x)
        return x