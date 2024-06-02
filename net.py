import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        # Define the feature extraction part of the network
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the classification part of the network
        self.fc1 = nn.Linear(32 * 4 * 4, 256)  # Adjusted size due to pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Feature extraction
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)

        x = F.leaky_relu(self.conv3(x))
        x = self.pool3(x)

        x = F.leaky_relu(self.conv4(x))
        x = self.pool4(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Classification
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def trainStep(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x)
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        return loss