import torch
import torchvision.datasets
from net import Net
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    print("running on: ", device)

    net = Net(num_classes=251)
    summary(net, (3, 64, 64))
    net.to(device)

    lossOvertime = []
    accuracyOvertime = []

    # Define data transformations pipeline
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((64, 64))
    ])

    trainSet = torchvision.datasets.ImageFolder(root='./Data/processedData/processed_train_set', transform=transforms)
    testSet = torchvision.datasets.ImageFolder(root='./Data/processedData/processed_test_set', transform=transforms)

    trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)
    testLoader = DataLoader(testSet, batch_size=64, shuffle=True, num_workers=2)

    epochs = 1

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in tqdm(enumerate(trainLoader, 0), total=len(trainLoader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            loss = net.trainStep(inputs, labels)
            running_loss += loss.item()

        lossOvertime.append(running_loss)


        # check accuracy on test set
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
        accuracyOvertime.append(accuracy)
        print(f"Epoch {epoch + 1}, loss: {running_loss}, accuracy: {accuracy}")

        # save model
        torch.save(net.state_dict(), f"model_{epoch}.pth")


if __name__ == '__main__':
    main()