import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer

    def forward(self, x):
        # Forward pass: input -> fc1 -> ReLU -> fc2
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
