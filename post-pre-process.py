import torchvision.datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# Function to calculate mean and std
def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0

    pbar = tqdm(total=len(loader), desc='Calculating', unit='frame')

    for images, _ in loader:
        # batch size (the last batch can have smaller size)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

        pbar.update(1)

    mean /= total_images_count
    std /= total_images_count

    return mean, std

def main():
    # Define data transformations pipeline
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])

    trainSet = torchvision.datasets.ImageFolder(root='./Data/processedData/processed_train_set', transform=transforms)

    trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=2)

    # Calculate mean and std
    mean, std = calculate_mean_std(trainLoader)
    print('Mean:', mean)
    print('Std:', std)

    # Mean: tensor([0.6388, 0.5446, 0.4452])
    # Std: tensor([0.2252, 0.2437, 0.2661])


if __name__ == '__main__':
    main()