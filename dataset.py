import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import itertools
from PIL import Image
from torch.optim.lr_scheduler import StepLR


class JigsawPuzzleDataset(Dataset):
    def __init__(self, dataset, grid_size=2):
        self.dataset = dataset
        self.grid_size = grid_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        image_pieces, correct_order = self._divide_image(image)
        shuffled_pieces, shuffled_order, indexes = self._shuffle_pieces(image_pieces, correct_order)

        # Create a new blank image of the correct size
        reconstructed_image = Image.new('RGB', image.size)

        # shuffle only 2 pieces and reconstruct the image
        piece_w, piece_h = image.size[0] // self.grid_size, image.size[1] // self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate the position of the piece in the reconstructed image
                pos = (j * piece_w, i * piece_h)
                # Paste the piece at the correct position
                reconstructed_image.paste(shuffled_pieces[i * self.grid_size + j], pos)

        # Convert the reconstructed image to a tensor
        reconstructed_image = transforms.ToTensor()(reconstructed_image)

        label = indexes

        # reoder them ascendent
        label = sorted(label)

        # piece_w, piece_h = image.size[0] // self.grid_size, image.size[1] // self.grid_size
        # for i in range(self.grid_size):
        #     for j in range(self.grid_size):
        #         # Calculate the position of the piece in the reconstructed image
        #         pos = (j * piece_w, i * piece_h)
        #         # Paste the piece at the correct position
        #         reconstructed_image.paste(shuffled_pieces[i * self.grid_size + j], pos)
        #
        # # Convert the reconstructed image to a tensor
        # reconstructed_image = transforms.ToTensor()(reconstructed_image)
        #
        # # One-hot encode the permutation index
        # permutation_index = self.permutations.index(tuple(shuffled_order))
        # label = torch.zeros(len(self.permutations), dtype=torch.float)
        # label[permutation_index] = 1.0

        return reconstructed_image, label

    def _divide_image(self, image):
        w, h = image.size
        piece_w, piece_h = w // self.grid_size, h // self.grid_size
        pieces = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                piece = image.crop((j * piece_w, i * piece_h, (j + 1) * piece_w, (i + 1) * piece_h))
                pieces.append(piece)

        correct_order = list(range(self.grid_size ** 2))
        return pieces, correct_order

    def _shuffle_pieces(self, pieces, correct_order):
        # shuffle only 2 pieces, draw 2 random numbers
        random_numbers = random.sample(range(0, 4), 2)
        shuffled_pieces = pieces.copy()
        shuffled_pieces[random_numbers[0]], shuffled_pieces[random_numbers[1]] = shuffled_pieces[random_numbers[1]], \
        shuffled_pieces[random_numbers[0]]

        shuffled_order = correct_order.copy()
        shuffled_order[random_numbers[0]], shuffled_order[random_numbers[1]] = shuffled_order[random_numbers[1]], \
        shuffled_order[random_numbers[0]]

        return shuffled_pieces, shuffled_order, random_numbers

        # shuffled_order = random.choice(self.permutations)
        # shuffled_pieces = [pieces[i] for i in shuffled_order]
        # return shuffled_pieces, shuffled_order