import torch
from torch.utils.data import DataLoader
from data.datatools import veri_dataset
from tools.utilities import sample_visualizer, accuracy as Acc
from tools.confusion_meter import ConfusionMeter
import os, sys
from tqdm import tqdm
import pdb

Orientation_labels = ['front', 'rear', 'left', 'left front', 'left rear', 'right', 'right front', 'right rear']


def test(args, net, epoch=None):
    """
    This is the function to test the trained key-point and orientation estimation model
    :param args: the object that encapsulates all the required settings
    :param net: the network to be tested
    :param epoch: the epoch number that this test is being done for
    :return: a dictionary that contains the result of the test
    """
    test_set = veri_dataset.VeriDataset(phase='test')

    test_loader = DataLoader(test_set, shuffle=False, batch_size=args.test_batch_size, num_workers=args.num_workers)

    if args.use_case == 'stage1':
        coarse_error = 0.0
        with torch.no_grad():
            with tqdm(total=len(test_loader), ncols=0, file=sys.stdout, desc='Stage 1 Evaluation...') as pbar:
                for i, in_batch in enumerate(test_loader):
                    image_in1, _, gt_heatmaps, _, _ = in_batch

                    if torch.cuda.is_available():
                        image_in1, gt_heatmaps = image_in1.cuda(), gt_heatmaps.cuda()

                    coarse_kp = net(image_in1)

                    if args.visualize:
                        sample_visualizer(coarse_kp[:, :20, :, :], gt_heatmaps[:, :20, :, :], image_in1)

                    coarse_error += Acc(coarse_kp, gt_heatmaps)
                    pbar.set_postfix(coarse_kp_dist_from_gt=coarse_error / (i + 1))
                    pbar.update()

        coarse_error = coarse_error / len(test_loader)
        message = 'Stage 1 KP estimation error is {0:.3f} pixels in 56 by 56 grid.'.format(coarse_error)
        return {'message': message, 'coarse_error': coarse_error}

    elif args.use_case == 'stage2':
        coarse_error, fine_error, orientation_accuracy, total, correct, = 0.0, 0.0, 0.0, 0, 0
        if args.phase == 'train':
            save_path = os.path.join(args.ckpt, 'epoch_{}'.format(epoch + 1) + '.png')
            orientation_cmf = ConfusionMeter(normalize=True, save_path=save_path, labels=Orientation_labels)
        with torch.no_grad():
            with tqdm(total=len(test_loader), ncols=0, file=sys.stdout, desc='Stage 2 Evaluation...') as pbar:
                for i, in_batch in enumerate(test_loader):
                    image_in1, image_in2, gt_heatmaps, _, gt_orientation_label = in_batch
                    if torch.cuda.is_available():
                        image_in1, image_in2, gt_heatmaps, gt_orientation_label = image_in1.cuda(),\
                                                                                  image_in2.cuda(),\
                                                                                  gt_heatmaps.cuda(),\
                                                                                  gt_orientation_label.cuda()
                    coarse_kp, fine_kp, orientation = net(image_in1, image_in2)
                    if args.visualize:
                        sample_visualizer(fine_kp[:, :20, :, :], gt_heatmaps[:, :20, :, :], image_in1)
                    _, predicted_orientation = torch.max(orientation.data, 1)
                    if args.phase == 'train':
                        orientation_cmf.update(predicted_orientation, gt_orientation_label.squeeze())
                    total += gt_orientation_label.size(0)
                    correct += (gt_orientation_label.squeeze() == predicted_orientation).sum().item()
                    coarse_error += Acc(coarse_kp, gt_heatmaps)
                    fine_error += Acc(fine_kp, gt_heatmaps)
                    pbar.set_postfix(fine_kp_dist_from_gt=fine_error / (i + 1),
                                     orientation_classification_accuracy=float(correct) / total * 100)
                    pbar.update()
        if not args.phase == 'test':
            orientation_cmf.save_confusion_matrix()

        coarse_error, fine_error, orientation_accuracy = \
            coarse_error / len(test_loader), \
            fine_error / len(test_loader), \
            float(correct) / total * 100

        message = 'Stages 1 & 2 KP estimation errors are {0:.3f} & {1:.3f} pixels in 56 by 56 grid & Orientation' \
                  ' classification accuracy is {2:.2f}%.'.format(coarse_error, fine_error, orientation_accuracy)

        return {'message': message,
                'coarse_error': coarse_error,
                'fine_error': fine_error,
                'orientation_accuracy': orientation_accuracy}