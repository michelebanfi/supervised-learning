import torchvision.datasets
from torch.utils.data import DataLoader

# Function to calculate mean and std
def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in loader:
        # batch size (the last batch can have smaller size)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

def main():
    # Define data transformations pipeline
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    trainSet = torchvision.datasets.ImageFolder(root='./Data/processedData/processed_train_set', transform=transforms)

    trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)

    # Calculate mean and std
    mean, std = calculate_mean_std(trainLoader)
    print('Mean:', mean)
    print('Std:', std)

    # Mean: tensor([0.6385, 0.5444, 0.4450])
    # Std: tensor([0.2262, 0.2446, 0.2658])


if __name__ == '__main__':
    main()