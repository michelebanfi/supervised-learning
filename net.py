import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes, size):
        super(Net, self).__init__()
        # Define the feature extraction part of the network
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        size /= 2
        # 122

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        size /= 2
        # 124

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        size /= 2
        # 124

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        size /= 2
        # 124

        convLayerNumber = 4
        kernels = [11, 5, 3, 3]
        paddings = [1, 1, 1, 1]
        poolingsStride = [2, 2, 2, 2]
        poolingsKernels = [2, 2, 2, 2]




        # Calculate the size of the input to the first fully connected layer
        # Input image size is (128, 128)
        # After first pooling: (128/2) = 64
        # After second pooling: (64/2) = 32
        # After third pooling: (32/2) = 16
        # After fourth pooling: (16/2) = 8

        # cast size to int
        size = int(size)

        fc1_input_size = 32 * size * size

        # Define the classification part of the network
        self.fc1 = nn.Linear(fc1_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
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

        # create a tensor for each class
        target = torch.zeros((y.size(0), 251))
        target[range(y.size(0)), y] = 1

        loss = self.criterion(out, target)

        loss.backward()
        self.optimizer.step()
        return loss
