import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Extends torch.utils.data.Dataset to our purposes
    
    We often simply pass tensors to pytorch using torch.utils.data.TensorData,
    but since our data is sitting in image files, it not quite so easy.
    """
    def __init__(self, transform=None):
        self.transform = transform
        self.n = self.get_n()
        self.c = self.get_c()
        self.sz = self.get_sz()

    def get_n(self):
        pass
    
    def get_c(self):
        pass

    def get_sz(self):
        pass

def sample_img(path, num_samples=1):
    # Sample from the given directory a number of images
    samples = np.random.choice(os.listdir(path), size=num_samples)
    for sample in samples:
        # Read each image and display
        img = plt.imread(f'{path}/{sample}')
        plt.figure()
        plt.imshow(img)