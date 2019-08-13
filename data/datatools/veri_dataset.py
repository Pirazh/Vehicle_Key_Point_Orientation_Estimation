from __future__ import print_function, division
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import scipy.ndimage as ndimage
from skimage import io
from tools.paths import paths
from . import transforms
import pdb


pose_flip_lr_dict = {'0': 0, '1': 1, '2': 5, '3': 6, '4': 7, '5': 2, '6': 3, '7': 4}


class VeriDataset(Dataset):
    """
    Veri Dataset class
    """
    def __init__(self, phase='train', flip_probability=0, rotate_probability=0):
        self.phase = phase
        self.flip_probability = flip_probability
        self.rotate_probability = rotate_probability
        self.struct = ndimage.generate_binary_structure(2, 1)

        if self.phase == 'train':
            txt_file = paths.VERI_KP_ANNOTATIONS_TRAINING_FILE
        elif self.phase == 'test':
            txt_file = paths.VERI_KP_ANNOTATIONS_TESTING_FILE
        else:
            raise NameError('Phase should be either "train" or "test"')
        assert (os.path.exists(txt_file))
        self.anno = [line.rstrip('\n') for line in open(txt_file)]

        # remove missing data files
        no_data = []
        for line in self.anno:
            if not os.path.isfile(os.path.join(paths.VERI_DATA_PATH, line.split(' ')[0])):
                no_data.append(line)
        for line in no_data:
            self.anno.remove(line)

        self.Normalize = transforms.Normalize()
        self.LRflip = transforms.LRFlip()
        self.Rescale = transforms.Rescale()
        self.ToTensor = transforms.ToTensor()
        self.Rotate = transforms.Rotate()
        self.key_point_distribution, self.pose_distribution = self._class_weights()

    def _class_weights(self):
        """
        Calculate the frequency of each class to help balance the training ot the model.
        :return: Inverse frequency of each pixel type class (20 key-points + background) and
                    each vehicle orientation class
        """
        txt_file = paths.VERI_KP_ANNOTATIONS_TRAINING_FILE
        anno = [line.rstrip('\n') for line in open(txt_file)]

        no_data = []
        for line in anno:
            if not os.path.isfile(os.path.join(paths.VERI_DATA_PATH, line.split(' ')[0])):
                no_data.append(line)
        for line in no_data:
            anno.remove(line)

        weights = np.zeros(21)
        pose_distribution = np.zeros(8)
        for line in anno:
            cnt = 0
            pose_distribution[int(line[-1])] += 1
            for i in range(0, 20):
                coordinate = line.split(' ')[2 * i + 1: 2 * i + 3]
                if int(coordinate[0]) > -1:
                    cnt += 1
                    weights[i] += 1
            weights[20] += 56 * 56 - cnt

        return torch.from_numpy((1 / weights) / (1 / weights).sum()),\
               torch.from_numpy((1 / pose_distribution) / (1 / pose_distribution).sum())

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, item):
        # load the image
        im_path = os.path.join(paths.VERI_DATA_PATH, self.anno[item].split(' ')[0])
        image = io.imread(im_path)
        image = image.astype(np.float)

        image = self.Normalize(image)
        image_in1, image_in2 = self.Rescale(image)

        H, W = image.shape[0], image.shape[1]

        pose = int(self.anno[item][-1])

        # load annotations
        keypoints = np.zeros([20, 2])

        for i in range(0, 20):
            keypoints[i] = [int(b) for b in self.anno[item].split(' ')[2 * i + 1: 2 * i + 3]]

        # Resize annotations to fit the modifed image
        keypoints[:, 0] = keypoints[:, 0] * (56 / W)
        keypoints[:, 1] = keypoints[:, 1] * (56 / H)

        if self.phase == 'train':

            # LR Flipping
            if np.random.rand() < self.flip_probability:
                image_in1, image_in2 = self.LRflip(image_in1), self.LRflip(image_in2)
                pose = pose_flip_lr_dict[str(pose)]
                keypoints[:, 0] = 56 - keypoints[:, 0]

            # Rotation
            if np.random.rand() < self.rotate_probability:
                angle = int(np.random.rand() * 10 - 5 / 2)
                image_in1, image_in2 = self.Rotate(image_in1, theta=angle), self.Rotate(image_in2, theta=angle)
                angle_radian = np.pi * angle / 180
                R = np.array([[np.cos(angle_radian), np.sin(angle_radian)],
                              [-np.sin(angle_radian), np.cos(angle_radian)]])
                keypoints = np.matmul(R, keypoints.T - 28).T + 28

        gt_heatmaps = np.zeros([21, 56, 56])
        pixel_class_label = np.ones([56, 56]) * 20

        for i, pt in enumerate(keypoints):
            if (55 >= pt[0] > 0) and (55 >= pt[1] > 0):
                gt_heatmaps[i][int(pt[1])][int(pt[0])] = 1
                pixel_class_label[int(pt[1])][int(pt[0])] = i

        # defining map corresponding to the background
        gt_heatmaps[20] = np.ones([56, 56])

        for i in range(0, 20):
            gt_heatmaps[20] = gt_heatmaps[20] - gt_heatmaps[i]
        """
        for i in range(20):
            gt_heatmaps[i] = \
                ndimage.binary_dilation(gt_heatmaps[i], structure=self.struct, iterations=1).astype(gt_heatmaps.dtype)
        """
        #pdb.set_trace()
        return self.ToTensor(image_in1.transpose(2, 0, 1)),\
               self.ToTensor(image_in2.transpose(2, 0, 1)), \
               torch.from_numpy(gt_heatmaps).float(), \
               torch.from_numpy(pixel_class_label).long(), \
               torch.tensor([pose])