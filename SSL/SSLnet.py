import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from net import Net

class NetWithRotationPrediction(Net):
    def __init__(self, num_classes, size):
        super(NetWithRotationPrediction, self).__init__(num_classes, size)
        self.rotation_fc = nn.Linear(self.fc1.in_features, 4)  # 4 classes for rotations

    def forward(self, x, predict_rotation=False):
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
        if predict_rotation:
            return self.rotation_fc(x)
        else:
            x = F.leaky_relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)