import torch
import torchvision.datasets

from net import Net
from torchsummary import summary
import torchvision.transforms as transforms

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if train_on_gpu else "cpu")
print(device)

net = Net()
summary(net, (1, 28, 28))
net.to(device)

lossOvertime = []
accuracyOvertime = []

# Define data transformations pipeline
transforms = transforms.Compose([
    transforms.ToTensor()
])

trainSet = torchvision.datasets.ImageFolder(root='./Data/processedData/processed_train_set', transform=transforms)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, shuffle=True, num_workers=2)

epochs = 10

# for epoch in range(epochs):
#     running_loss = 0.0
#     for i, data in enumerate(trainLoader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         loss = net.trainStep(inputs, labels)
#         running_loss += loss
#     lossOvertime.append(running_loss)
#     print(f"Epoch {epoch+1}, loss: {running_loss}")