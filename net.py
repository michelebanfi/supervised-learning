import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Net(nn.Module):
    def __init__(self, num_classes, size):

        # list of the conv layers parameters
        convLayerNumber = 7
        kernels = [11, 7, 7, 5, 5, 3, 3]
        paddings = [1, 1, 1, 1, 1, 1, 1]
        poolingsStride = [2, 0, 2, 0, 2, 0, 2]
        poolingsKernels = [2, 0, 2, 0, 2, 0, 2]
        filters = [8, 16, 16, 32, 32, 64, 64]

        super(Net, self).__init__()
        # Define the feature extraction part of the network
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=kernels[0], padding=paddings[0])
        self.pool1 = nn.MaxPool2d(kernel_size=poolingsKernels[0], stride=poolingsStride[0])

        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=kernels[1], padding=paddings[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=kernels[2], padding=paddings[2])
        self.pool2 = nn.MaxPool2d(kernel_size=poolingsKernels[2], stride=poolingsStride[2])

        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=kernels[3], padding=paddings[3])
        self.conv5 = nn.Conv2d(filters[3], filters[4], kernel_size=kernels[4], padding=paddings[4])
        self.pool3 = nn.MaxPool2d(kernel_size=poolingsKernels[4], stride=poolingsStride[4])

        self.conv6 = nn.Conv2d(filters[4], filters[5], kernel_size=kernels[5], padding=paddings[5])
        self.conv7 = nn.Conv2d(filters[5], filters[6], kernel_size=kernels[6], padding=paddings[6])
        self.pool4 = nn.MaxPool2d(kernel_size=poolingsKernels[6], stride=poolingsStride[6])

        # now we want to calculate the final dimension of all the conv layers
        for i in range(convLayerNumber):
            size = (size - kernels[i] + 2 * paddings[i]) / 1 + 1
            if (poolingsKernels[i] != 0):
                size = int((size - poolingsKernels[i]) / poolingsStride[i] + 1)

        fc1_input_size = filters[6] * size * size

        print("First layer size: ", fc1_input_size)

        # Define the classification part of the network
        self.fc1 = nn.Linear(fc1_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Feature extraction
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)

        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.pool2(x)

        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = self.pool3(x)

        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = self.pool4(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Classification
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def trainStep(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x)

        # create a tensor for each class
        target = torch.zeros((y.size(0), 251))
        target[range(y.size(0)), y] = 1

        loss = self.criterion(out, y)

        loss.backward()
        self.optimizer.step()
        return loss
