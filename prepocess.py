import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")
print("running on:", device)


# class list pre processing
def add_string(file, string):
    with open(file, 'r') as original: data = original.read()
    if data.startswith(string): return
    with open(file, 'w') as modified: modified.write(string + data)


add_string('Data/annot/train_info.csv', 'image,class\n')
add_string('Data/annot/val_info.csv', 'image,class\n')

# Directory:
trainDirectory = 'Data/train_set'
testDirectory = 'Data/val_set'

processedTrainDirectory = 'Data/processedData/processed_train_set'
processedTestDirectory = 'Data/processedData/processed_test_set'
processedValDirectory = 'Data/processedData/processed_val_set'

# Creation of the directory if they dont exist
os.makedirs(processedTrainDirectory, exist_ok=True)
os.makedirs(processedTestDirectory, exist_ok=True)
os.makedirs(processedValDirectory, exist_ok=True)

# load images classes from "train_info.csv"
trainInfo = pd.read_csv('Data/annot/train_info.csv')
testInfo = pd.read_csv('Data/annot/val_info.csv')

trainClasses = trainInfo['class'].unique()
testClasses = testInfo['class'].unique()

# create folders for each train class
for trainClass in trainClasses:
    os.makedirs(os.path.join(processedTrainDirectory, str(trainClass)), exist_ok=True)

# create folders for each test class
for testClass in testClasses:
    os.makedirs(os.path.join(processedTestDirectory, str(testClass)), exist_ok=True)

# create folders for each validation class (using the train classes)
for valClass in trainClasses:
    os.makedirs(os.path.join(processedValDirectory, str(valClass)), exist_ok=True)


def process_images(input_dir, output_dir, set_type):
    image_path_pattern = os.path.join(input_dir, "*.jpg")
    image_paths = glob.glob(image_path_pattern)

    pbar = tqdm(total=len(image_paths), desc='Processing', unit='frame')

    for image_path in image_paths:
        # get image class
        if set_type == "train":
            imageClass = trainInfo[trainInfo['image'] == os.path.basename(image_path)]['class'].values[0]
        else:
            imageClass = testInfo[testInfo['image'] == os.path.basename(image_path)]['class'].values[0]

        # path to save the images
        base_filename = os.path.basename(image_path)
        processed_image_path = os.path.join(output_dir, str(imageClass), base_filename)

        pbar.update(1)


def valSet():
    # take 3% of the train set and create a vaidation set
    image_path_pattern = os.path.join(trainDirectory, "*.jpg")
    image_paths = glob.glob(image_path_pattern)

    # for the validation set, draw 3% of the train set. But be sure that at least one image of each class is in the validation set
    val_set = []
    for trainClass in trainClasses:
        matching_rows = trainInfo[trainInfo['class'] == trainClass]
        image = matching_rows['image'].values[0]
        image_path = os.path.join(trainDirectory, image)
        val_set.append(image_path)

    # draw the rest of the images
    rand_choice = np.random.choice(image_paths, int(len(image_paths) * 0.03) - len(val_set), replace=False)
    val_set.extend(rand_choice)

    pbar = tqdm(total=len(val_set), desc='Processing', unit='frame')

    # path to save the images
    for image_path in val_set:
        matching_rows = trainInfo[trainInfo['image'] == os.path.basename(image_path)]

        imageClass = matching_rows['class'].values[0]
        base_filename = os.path.basename(image_path)

        processed_image_path = os.path.join(processedValDirectory, str(imageClass), base_filename)
        try:
            # move the images to the test set
            os.rename(image_path, processed_image_path)
        except:
            print("Error moving the image")
        pbar.update(1)

# Creation of the validation set
print("Creating validation set and processing it")
valSet()

print("Processing train set")
process_images(trainDirectory, processedTrainDirectory, "train")

print("Processing test set")
process_images(testDirectory, processedTestDirectory, "test")
