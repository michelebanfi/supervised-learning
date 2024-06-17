# iFood 2019 ~ Food Classification Dataset

## Structure of the project

Inside the `notebook.ipynb` file the code used to train test and validate the networks.

In order to properly create the Dataset a preprocessing step is needed. Inside
the `preprocess.py` all the code needed is present. The file assumes that the orignal data
from the competition is present inside a folder called `Data`. After a preprocessing step a new
file called `post-pre-process.py` will perform calculate mean and std on the training images.
