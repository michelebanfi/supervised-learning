import torch
import cv2
import glob  # Import library for finding all files matching a pattern
from PIL import Image  # Import library for image processing
import os  # Import library for operating system functionalities

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if train_on_gpu else "cpu")
print(device)

trainDirectory = 'Data/train_set'
testDirectory = 'Data/val_set'

processedTrainDirectory = 'Data/processedData/processed_train_set'
processedTestDirectory = 'Data/processedData/processed_test_set'

# Define a path pattern to search for all jpg images within subdirectories of "/content/imdb_crop"
image_path_pattern = trainDirectory +"/*.jpg"
image_paths = glob.glob(image_path_pattern)

for image_path in image_paths:
    image = cv2.imread(image_path)

    if (len(image.shape) != 3):
        print("Not Colored")
    #image = cv2.cvtColor(image, cv2)

    image = cv2.resize(image, (256, 256))

    # save the image
    cv2.imwrite(processedTrainDirectory + image_path)
