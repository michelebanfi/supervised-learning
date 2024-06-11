import torch
from torch.utils.data import Dataset
import random
import itertools


class JigsawPuzzleDataset(Dataset):
    def __init__(self, dataset, grid_size=2):
        self.dataset = dataset
        self.grid_size = grid_size
        self.permutations = self._generate_permutations(grid_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        image_pieces, correct_order = self._divide_image(image)
        shuffled_pieces, shuffled_order = self._shuffle_pieces(image_pieces, correct_order)

        # Create a new blank image of the correct size
        reconstructed_image = torch.zeros_like(image)

        piece_w, piece_h = image.shape[1] // self.grid_size, image.shape[2] // self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate the position of the piece in the reconstructed image
                pos = (j * piece_w, i * piece_h)
                # Paste the piece at the correct position
                reconstructed_image[:, pos[1]:pos[1]+piece_h, pos[0]:pos[0]+piece_w] = shuffled_pieces[i * self.grid_size + j]

        # One-hot encode the permutation index
        permutation_index = self.permutations.index(tuple(shuffled_order))
        label = torch.zeros(len(self.permutations), dtype=torch.float)
        label[permutation_index] = 1.0

        return reconstructed_image, label

    def _divide_image(self, image):
        pieces = []

        piece_w, piece_h = image.shape[1] // self.grid_size, image.shape[2] // self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                piece = image[:, i*piece_h:(i+1)*piece_h, j*piece_w:(j+1)*piece_w]
                pieces.append(piece)

        correct_order = list(range(self.grid_size ** 2))
        return pieces, correct_order

    def _shuffle_pieces(self, pieces, correct_order):
        shuffled_order = random.choice(self.permutations)
        shuffled_pieces = [pieces[i] for i in shuffled_order]
        return shuffled_pieces, shuffled_order

    def _generate_permutations(self, grid_size):
        indices = list(range(grid_size ** 2))
        permutations = list(itertools.permutations(indices))
        return permutations