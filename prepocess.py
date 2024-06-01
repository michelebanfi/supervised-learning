import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if train_on_gpu else "cpu")
print(device)

# Directory:
trainDirectory = 'Data/train_set'
testDirectory = 'Data/val_set'


processedTrainDirectory = 'Data/processedData/processed_train_set'
processedTestDirectory = 'Data/processedData/processed_test_set'

# Creation of the directory if they dont exist
os.makedirs(processedTrainDirectory, exist_ok=True)
os.makedirs(processedTestDirectory, exist_ok=True)

def process_images(input_dir, output_dir):
    image_path_pattern = os.path.join(input_dir, "*.jpg")
    image_paths = glob.glob(image_path_pattern)

    for image_path in image_paths:
        image = cv2.imread(image_path)

        # Check if the image is in RBG format
        if (len(image.shape) != 3):
            print("is not RGB")



        # Resize of the images to a common size
        image = cv2.resize(image, (256, 256))

    # path to save the images
        base_filename = os.path.basename(image_path)
        processed_image_path = os.path.join(output_dir, base_filename)

    # saving the images
        success = cv2.imwrite(processed_image_path, image)
        if success:
            print(f"Saved processed image to: {processed_image_path}")
        else:
            print(f"Failed to save image: {processed_image_path}")


    # Process of the images in the train set and the test set
process_images(trainDirectory, processedTrainDirectory)
process_images(testDirectory, processedTestDirectory)