import torch

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
