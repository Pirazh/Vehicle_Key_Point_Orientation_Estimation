import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

KP_labels = ['left-front wheel', 'left-back wheel', 'right-front wheel', 'right-back wheel', 'right fog lamp',
             'left fog lamp', 'right headlight', 'left headlight', 'front auto logo', 'front license plate',
             'left rear-view mirror', 'right rear-view mirror', 'right-front corner of vehicle top',
             'left-front corner of vehicle top', 'left-back corner of vehicle top', 'right-back corner of vehicle top',
             'left rear lamp', 'right rear lamp', 'rear auto logo', 'rear license plate']


class Chronometer:
    """
    Chronometer class to time the code
    """
    def __init__(self):
        self.elapsed = 0
        self.start = 0
        self.end = 0

    def set(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()
        self.elapsed = (self.end - self.start)

    def reset(self):
        self.start, self.end, self.elapsed = 0, 0, 0


def save_checkpoint(state, is_best, filename='./checkpoint/checkpoint.pth.tar',
                    best_filename='./checkpoint/model_best.pth.tar'):
    """
    Save trained model
    :param state: the state dictionary to be saved
    :param is_best: boolian to show if this is the best checkpoint so far
    :param filename: label of the current checkpoint
    :param best_filename: label of the so far best checkpoint
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def sample_visualizer(outputs, maps, inputs):
    """
    Randomly visualize the estimated key-points and their respective ground-truth maps
    :param outputs: the estimated key-points
    :param maps: the ground-truth maps
    :param inputs: the tensor containing the normalized image data
    :return: visualize the heatmaps
    """
    rand = np.random.randint(0, outputs.shape[0])
    map_out, map1, sample_in = outputs[rand], maps[rand], inputs[rand]

    plt.figure(2)
    plt.imshow(sample_in.transpose(0, 2).transpose(0, 1).cpu().numpy())
    plt.pause(.05)
    map_out = map_out.cpu().numpy()
    map1 = map1.cpu().numpy()

    plt.figure(1)
    for i in range(0, 20):
        plt.subplot(4, 10, i + 1)
        plt.imshow(map_out[i] / map_out.sum())
        plt.xlabel(KP_labels[i], fontsize=5)
        plt.xticks()
        plt.subplot(4, 10, 21 + i)
        plt.imshow(map1[i])
        plt.xlabel(KP_labels[i], fontsize=5)

    plt.pause(.05)
    plt.draw()


def get_preds(heatmaps):
    """
    Get the coordinates of
    :param heatmaps: heatmaps
    :return: coordinate of hottest points
    """
    assert heatmaps.dim() == 4, 'Score maps should be 4-dim Batch, Channel, Heigth, Width'

    maxval, idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), 2)
    maxval = maxval.view(heatmaps.size(0), heatmaps.size(1), 1)
    idx = idx.view(heatmaps.size(0), heatmaps.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % heatmaps.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / heatmaps.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def calc_dists(preds, target):
    """
    Calculate the average distance from predictions to their ground-truth
    :param preds: predicted coordinates of key-points from estimations
    :param target: predicted coordionates of key-point from ground-truth maps
    :return: the average distance
    """
    preds = preds.float()
    target = target.float()
    cnt = 0
    dists = 0
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                dists += torch.dist(preds[n, c, :], target[n, c, :])
                cnt += 1
    dists = dists / cnt
    return dists


def accuracy(output, target):
    """
    Calculate the accuracy of predicted key-points with respect to visible key-points in ground-truth
    :param output: the estimated key-points of shape B * 21 (or 20) * 56 * 56
    :param target: the gt-key-points of shape B * 21 * 56 * 56
    :return: the average distance of the hottest point from its ground-truth
    """
    preds = get_preds(output[:, :20, :, :])
    gts = get_preds(target[:, :20, :, :])
    dists = calc_dists(preds, gts)
    return dists.item()


