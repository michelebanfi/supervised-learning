import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import itertools
from PIL import Image

class JigsawPuzzleDataset(Dataset):
    def __init__(self, dataset, grid_size=2):
        self.dataset = dataset
        self.grid_size = grid_size
        self.permutations = self._generate_permutations(grid_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        image_pieces, correct_order = self._divide_image(image)
        shuffled_pieces, shuffled_order = self._shuffle_pieces(image_pieces, correct_order)

        # Create a new blank image of the correct size
        reconstructed_image = torch.zeros_like(image)

        piece_w, piece_h = image.shape[1] // self.grid_size, image.shape[2] // self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate the position of the piece in the reconstructed image
                pos = (j * piece_w, i * piece_h)
                # Paste the piece at the correct position
                reconstructed_image[:, pos[1]:pos[1]+piece_h, pos[0]:pos[0]+piece_w] = shuffled_pieces[i * self.grid_size + j]

        # One-hot encode the permutation index
        permutation_index = self.permutations.index(tuple(shuffled_order))
        label = torch.zeros(len(self.permutations), dtype=torch.float)
        label[permutation_index] = 1.0

        return reconstructed_image, label

    def _divide_image(self, image):
        pieces = []

        piece_w, piece_h = image.shape[1] // self.grid_size, image.shape[2] // self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                piece = image[:, i*piece_h:(i+1)*piece_h, j*piece_w:(j+1)*piece_w]
                pieces.append(piece)

        correct_order = list(range(self.grid_size ** 2))
        return pieces, correct_order

    def _shuffle_pieces(self, pieces, correct_order):
        shuffled_order = random.choice(self.permutations)
        shuffled_pieces = [pieces[i] for i in shuffled_order]
        return shuffled_pieces, shuffled_order

    def _generate_permutations(self, grid_size):
        indices = list(range(grid_size ** 2))
        permutations = list(itertools.permutations(indices))
        return permutations

def main():
    mean = [0.6388, 0.5446, 0.4452]
    std = [0.2252, 0.2437, 0.2661]

    class Net(nn.Module):
        def __init__(self, num_classes, size):

            # list of the conv layers parameters
            convLayerNumber = 7
            kernels = [11, 7, 7, 5, 5, 3, 3]
            paddings = [6, 4, 4, 3, 3, 2, 2]
            poolingsStride = [2, 0, 2, 0, 2, 0, 2]
            poolingsKernels = [4, 0, 2, 0, 2, 0, 2]
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
    class NetWithJigSawPrediction(Net):
        def __init__(self, num_classes, size):
            super(NetWithJigSawPrediction, self).__init__(num_classes, size)
            self.rotation_fc = nn.Linear(self.fc1.in_features, 24)

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

    trainSet = datasets.ImageFolder(root='./Data/processedData/processed_train_set', transform=transform)
    testSet = datasets.ImageFolder(root='./Data/processedData/processed_test_set', transform=transform)
    valSet = datasets.ImageFolder(root='./Data/processedData/processed_val_set', transform=transform)
    rotation_dataset = JigsawPuzzleDataset(trainSet)
    rotation_datasetTest = JigsawPuzzleDataset(testSet)
    rotation_datasetVal = JigsawPuzzleDataset(valSet)

    batch_size = 64
    rotation_loader = DataLoader(rotation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    rotation_loaderTest = DataLoader(rotation_datasetTest, batch_size=batch_size, shuffle=True, num_workers=1)
    rotation_loaderVal = DataLoader(rotation_datasetVal, batch_size=batch_size, shuffle=True, num_workers=1)

    print("Validation:", len(rotation_loader.dataset))
    print("Training:", len(rotation_loaderTest.dataset))
    print("Test:", len(rotation_loaderVal.dataset))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Initialize the SSL model
    ssl_model = NetWithJigSawPrediction(num_classes=251, size=128)
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
                labels = torch.argmax(labels, dim=1)
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

    classification_dataset = datasets.ImageFolder(root="./Data/processedData/processed_train_set",
                                                  transform=classification_transform)
    classification_datasetTest = datasets.ImageFolder(root="./Data/processedData/processed_test_set",
                                                      transform=classification_transform)
    classification_datasetVal = datasets.ImageFolder(root="./Data/processedData/processed_val_set",
                                                     transform=classification_transform)

    classification_loader = DataLoader(classification_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    classification_loaderTest = DataLoader(classification_datasetTest, batch_size=batch_size, shuffle=True,
                                           num_workers=1)
    classification_loaderVal = DataLoader(classification_datasetVal, batch_size=batch_size, shuffle=True, num_workers=1)

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

if __name__ == '__main__':
    main()