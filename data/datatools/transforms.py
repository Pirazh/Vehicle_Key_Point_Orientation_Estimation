import torch
from skimage import transform
import numpy as np
from tools import paths


class ToTensor(object):
    """
    The object to transform numpy arrays to torch tensors
    """
    def __call__(self, image):
        return torch.from_numpy(image).float()


class Rescale(object):
    """
    The object to rescale loaded images to desired output size
    """
    def __init__(self, input_size1=(224, 224), input_size2=(56, 56)):
        self.input_size1 = input_size1
        self.input_size2 = input_size2

    def __call__(self, image):

        image_in2 = transform.resize(image, self.input_size2)
        image_in1 = transform.resize(image, self.input_size1)

        return image_in1, image_in2


class Normalize(object):
    """
    The object to normalize the images based on the Veri-776 training set mean and standard deviation
    """
    def __init__(self, dataset_mean_std=paths.paths.VERI_MEAN_STD_FILE):
        mean_std = torch.load(dataset_mean_std)
        self.mean = mean_std['mean'].numpy()
        self.std = mean_std['std'].numpy()

    def __call__(self, image):
        for j in range(3):
            image[:, :, j] = (image[:, :, j] - self.mean[j]) / self.std[j]
        return image


class Rotate(object):
    """
    The object to rotate the input image with a desired angle
    """
    def __init__(self):
        pass

    def __call__(self, image, theta=0):
        image = np.round(transform.rotate(image, theta, preserve_range=True))
        return image


class LRFlip(object):
    """
    The object to horizontally mirror the input image
    """
    def __init__(self):
        pass

    def __call__(self, image):
        return np.fliplr(image).copy()
