import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from SSL.JigSawDataset import JigsawPuzzleDataset
from SSL.SSLnet import NetWithJigSawPrediction

def main():
    mean = [0.6388, 0.5446, 0.4452]
    std = [0.2252, 0.2437, 0.2661]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to a standard size
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    trainSet = datasets.ImageFolder(root='./Data/processedData/processed_train_set', transform=transform)
    testSet = datasets.ImageFolder(root='./Data/processedData/processed_test_set', transform=transform)
    valSet = datasets.ImageFolder(root='./Data/processedData/processed_val_set', transform=transform)
    jigsaw_dataset = JigsawPuzzleDataset(trainSet)
    jigsaw_datasetTest = JigsawPuzzleDataset(testSet)
    jigsaw_datasetVal = JigsawPuzzleDataset(valSet)

    batch_size = 64
    rotation_loader = DataLoader(jigsaw_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    rotation_loaderTest = DataLoader(jigsaw_datasetTest, batch_size=batch_size, shuffle=True, num_workers=1)
    rotation_loaderVal = DataLoader(jigsaw_datasetVal, batch_size=batch_size, shuffle=True, num_workers=1)

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