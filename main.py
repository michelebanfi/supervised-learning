import torch
import torchvision.datasets
from net import Net
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

def main(loadPreTrained: bool):
    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    print("running on: ", device)

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

    size = 128
    mean = [0.6385, 0.5444, 0.4450]
    std = [0.2262, 0.2446, 0.2658]
    net = Net(num_classes=251, size=size)
    summary(net, (3, size, size))

    # load a .pth file into the model in order to start from a pre-trained model
    if loadPreTrained:
        net.load_state_dict(torch.load('Models/128- model_2.pth'))
        net.eval()


    net.to(device)

    lossOvertime = []
    accuracyOvertime = []

    # Define data transformations pipeline
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])

    trainSet = torchvision.datasets.ImageFolder(root='./Data/processedData/processed_train_set', transform=transforms)
    testSet = torchvision.datasets.ImageFolder(root='./Data/processedData/processed_test_set', transform=transforms)
    valSet = torchvision.datasets.ImageFolder(root='./Data/processedData/processed_val_set', transform=transforms)

    trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)
    testLoader = DataLoader(testSet, batch_size=64, shuffle=True, num_workers=2)
    valLoader = DataLoader(valSet, batch_size=64, shuffle=True, num_workers=2)

    epochs = 5

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in tqdm(enumerate(trainLoader, 0), total=len(trainLoader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            loss = net.trainStep(inputs, labels)
            running_loss += loss.item()

        lossOvertime.append(round(running_loss/len(trainLoader), 2))

        # check accuracy on val set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(valLoader, total=len(valLoader)):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        accuracy = round(accuracy, 2)
        accuracyOvertime.append(accuracy)
        print(f"Epoch {epoch + 1}, loss: {round(running_loss/len(trainLoader), 2)}, accuracy: {accuracy}")

        title = ""
        if loadPreTrained:
            title = "-Pre-trained model"

        # save model
        torch.save(net.state_dict(), f"Models/{size}-model_{epoch}{title}.pth")

    print('Finished Training')
    print(lossOvertime)
    print(accuracyOvertime)

    # plot loss and accuracy in separate graphs
    plt.plot(lossOvertime)
    plt.savefig('Media/loss.png')
    plt.close()

    plt.plot(accuracyOvertime)
    plt.savefig('Media/accuracy.png')
    plt.close()

    print("Starting testing")

    # validate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation accuracy: {accuracy}")

    # calculate the F1 score
    print("Starting F1 score calculation")
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in valLoader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true += labels.tolist()
            y_pred += predicted.tolist()

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 score: {f1}")



if __name__ == '__main__':
    loadPreTrained = False

    main(loadPreTrained)