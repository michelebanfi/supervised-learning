import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt


mean = [0.6388, 0.5446, 0.4452]
std = [0.2252, 0.2437, 0.2661]

class Net(nn.Module):
    def __init__(self, num_classes, size):

        # list of the conv layers parameters
        convLayerNumber = 7
        kernels = [11, 7, 7, 5, 5, 3, 3]
        paddings = [1, 1, 1, 1, 1, 1, 1]
        poolingsStride = [2, 0, 2, 0, 2, 0, 2]
        poolingsKernels = [2, 0, 2, 0, 2, 0, 2]
        filters = [8, 16, 16, 32, 64, 64, 128]

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

        self.dropout = nn.Dropout(0.15)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        # self.criterion = nn.CrossEntropyLoss()

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

class RotationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.rotation_angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        rotation_label = np.random.randint(4)
        rotation_angle = self.rotation_angles[rotation_label]
        rotated_image = transforms.functional.rotate(image, rotation_angle)
        return rotated_image, rotation_label

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

        if predict_rotation:
            return self.rotation_fc(x)
        else:
            x = F.leaky_relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


trainSet = datasets.ImageFolder(root='/kaggle/input/supervised/processedData/processed_train_set', transform=transform)
testSet = datasets.ImageFolder(root='/kaggle/input/supervised/processedData/processed_test_set', transform=transform)
valSet = datasets.ImageFolder(root='/kaggle/input/supervised/processedData/processed_val_set', transform=transform)
rotation_dataset = RotationDataset(trainSet)
rotation_datasetTest = RotationDataset(testSet)
rotation_datasetVal = RotationDataset(valSet)

batch_size = 64
rotation_loader = DataLoader(rotation_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
rotation_loaderTest = DataLoader(rotation_datasetTest, batch_size=batch_size, shuffle=True, num_workers=8)
rotation_loaderVal = DataLoader(rotation_datasetTest, batch_size=batch_size, shuffle=True, num_workers=8)

print("Validation:", len(rotation_loader.dataset))
print("Training:", len(rotation_loaderTest.dataset))
print("Test:", len(rotation_loaderVal.dataset))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

# Initialize the SSL model
ssl_model = NetWithRotationPrediction(num_classes=251, size=128)
ssl_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)

# define a learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

num_epochs = 10

lossOvertime = []
accuracyOvertime = []

ssl_model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in tqdm(rotation_loader, total=len(rotation_loader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = ssl_model(images, predict_rotation=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    lossOvertime.append(round(running_loss / len(rotation_loader), 2))

    # put the model in evaluation mode
    ssl_model.eval()

    # check accuracy on val set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(rotation_loaderVal, total=len(rotation_loaderVal)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = ssl_model(images, predict_rotation=True)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    accuracy = round(accuracy, 2)
    accuracyOvertime.append(accuracy)
    print(f"Epoch {epoch + 1}, loss: {round(running_loss / len(rotation_loader), 2)}, accuracy: {accuracy}")

    # put the model back in training mode
    ssl_model.train()

    # save the model after each epoch
    torch.save(ssl_model.state_dict(), f"ssl_model_epoch_{epoch + 1}.pth")

print("Finished SSL Training")
print(lossOvertime)
print(accuracyOvertime)

# plot loss and accuracy in separate graphs
plt.plot(lossOvertime)
plt.savefig('Media/SSL/loss.png')
plt.close()

plt.plot(accuracyOvertime)
plt.savefig('Media/SSL/accuracy.png')
plt.close()

# start testing
print("Starting testing")

# put the model in evaluation mode
ssl_model.eval()

# validate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in rotation_loaderTest:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = ssl_model(images, predict_rotation=True)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Testings accuracy: {accuracy}")

classification_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

classification_dataset = datasets.ImageFolder(root="/kaggle/input/supervised/processedData/processed_train_set", transform=classification_transform)
classification_datasetTest = datasets.ImageFolder(root="/kaggle/input/supervised/processedData/processed_test_set", transform=classification_transform)
classification_datasetVal = datasets.ImageFolder(root="/kaggle/input/supervised/processedData/processed_val_set", transform=classification_transform)

classification_loader = DataLoader(classification_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
classification_loaderTest = DataLoader(classification_datasetTest, batch_size=batch_size, shuffle=True, num_workers=8)
classification_loaderVal = DataLoader(classification_datasetVal, batch_size=batch_size, shuffle=True, num_workers=8)

# Fine-tune the SSL model for classification
ssl_model.fc2 = nn.Linear(ssl_model.fc2.in_features, 251)  # Update the final layer for 251 classes
ssl_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ssl_model.parameters(), lr=0.001)

# define a learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

num_epochs = 10
lossOvertime = []
accuracyOvertime = []

ssl_model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in tqdm(classification_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = ssl_model(images, predict_rotation=False)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    lossOvertime.append(round(running_loss / len(classification_loader), 2))

    # put the model in evaluation mode
    ssl_model.eval()

    # check accuracy on val set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(classification_loaderVal, total=len(classification_loaderVal)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = ssl_model(images, predict_rotation=False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    accuracy = round(accuracy, 2)
    accuracyOvertime.append(accuracy)
    print(f"Epoch {epoch + 1}, loss: {round(running_loss / len(classification_loader), 2)}, accuracy: {accuracy}")

    # put the model back in training mode
    ssl_model.train()

    # save the model after each epoch
    torch.save(ssl_model.state_dict(), f"ssl_model_classification_epoch_{epoch + 1}.pth")

print("Finished Classification Training")
print(lossOvertime)
print(accuracyOvertime)

# plot loss and accuracy in separate graphs
plt.plot(lossOvertime)
plt.savefig('Media/SSL/classification_loss.png')
plt.close()

plt.plot(accuracyOvertime)
plt.savefig('Media/SSL/classification_accuracy.png')
plt.close()

# start testing
print("Starting testing")

# put the model in evaluation mode
ssl_model.eval()

# validate the model on the test set
correct = 0
total = 0

with torch.no_grad():
    for data in classification_loaderTest:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = ssl_model(images, predict_rotation=False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Testings accuracy: {accuracy}")