import time
import torch
import torch.nn as nn
from tools.utilities import save_checkpoint, Chronometer
from data.datatools import veri_dataset
from torch.utils.data import DataLoader
import os, sys
from tqdm import tqdm
from tools import test


def train(args, net):
    # Defininig Training Set and Data Loader
    train_set = veri_dataset.VeriDataset(phase=args.phase,
                                         rotate_probability=args.rotate_probability,
                                         flip_probability=args.flip_probability)
    train_loader = DataLoader(train_set,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    # Train Stage 1 for Coarse Heatmap Generation Using Pixel Based Classification
    if args.use_case == 'stage1':
        params = net.module.parameters() if args.mGPU else net.parameters()
        Heatmap_criterion = nn.CrossEntropyLoss(train_set.key_point_distribution.float().cuda())

        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        timer = Chronometer()
        start_epoch = args.start_epoch

        if not os.path.isdir(args.ckpt):
            os.mkdir(args.ckpt)

        if args.resume:
            checkpoint = torch.load(args.resumed_ckpt)
            start_epoch = checkpoint['epoch']

        best_error = 1000
        timer.set()
        # Initiate Logger
        with open(args.ckpt + '/logger.txt', 'w+') as f:
            f.write('Training Session on ' + time.strftime("%Y%m%d-%H") + '\n')
            # Write Used Arguments
            f.write('Used Arguments:\n')
            print('Used Arguments:')
            for key in args.__dict__.keys():
                f.write(key + ':{}\n'.format(args.__dict__[key]))
                print(key + ':{}'.format(args.__dict__[key]))

            # Training Loop
            for epoch in range(start_epoch, args.epochs):
                f.write('Epoch: {}'.format(epoch + 1))
                epoch_train_loss, is_best = 0.0, False
                with tqdm(total=len(train_loader), ncols=0, file=sys.stdout,
                          desc='Epoch: {}'.format(epoch + 1)) as pbar:

                    for i, in_batch in enumerate(train_loader):
                        optimizer.zero_grad()
                        image_in1, _, _, gt_pixel_label, _ = in_batch

                        if torch.cuda.is_available():
                            image_in1, gt_pixel_label = image_in1.cuda(), gt_pixel_label.cuda()

                        coarse_kp = net(image_in1)
                        loss = Heatmap_criterion(coarse_kp, gt_pixel_label)

                        epoch_train_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix(coarse_Kp=loss.item())
                        pbar.update()

                epoch_train_loss = epoch_train_loss / len(train_loader)
                # save the checkpoint
                save_checkpoint(
                    {'epoch': epoch + 1, 'net_state_dict': net.module.state_dict() if args.mGPU else net.state_dict()},
                    is_best, filename=os.path.join(args.ckpt, 'checkpoint.pth.tar'),
                    best_filename=os.path.join(args.ckpt, 'best_checkpoint.pth.tar'))

                f.write(', Average Training Loss: {} '.format(epoch_train_loss))
                print('Average Epoch Loss = {}'.format(epoch_train_loss))

                # Check Error of the Trained Model on test set
                if epoch % args.test_every_n_epoch == args.test_every_n_epoch - 1:
                    print('Testing the network...')
                    net.eval()
                    output = test.test(args, net, epoch)
                    print(output['message'])
                    if output['coarse_error'] < best_error:
                        best_error = output['coarse_error']
                        is_best = True
                    # save the checkpoint as best checkpoint so far
                    save_checkpoint(
                        {'epoch': epoch + 1,
                         'net_state_dict': net.module.state_dict() if args.mGPU else net.state_dict()},
                        is_best, filename=os.path.join(args.ckpt, 'checkpoint.pth.tar'),
                        best_filename=os.path.join(args.ckpt, 'best_checkpoint.pth.tar'))
                    f.write('\n')
                    f.write(output['message'])
                    f.write('\n')
                    net.train()

            timer.stop()
            f.write('Finished Trainig Session after {0} Epochs & {1} hours & {2} minutes, '
                    'Best coarse error Achieved: {3:.2f} pixel in 56 by 56 grid \n'
                    .format(args.epochs - start_epoch, int(timer.elapsed / 3600),
                            int((timer.elapsed % 3600) / 60), best_error))
            f.close()

        print('Finished Trainig Session after {0} Epochs & {1} hours & {2} minutes, '
                    'Best coarse error Achieved: {3:.2f} pixel in 56 by 56 grid \n'
                    .format(args.epochs - start_epoch, int(timer.elapsed / 3600),
                            int((timer.elapsed % 3600) / 60), best_error))

    # Train Stage 2 for Foarse Heatmap Regression and Orientation Estimation
    elif args.use_case == 'stage2':
        params = net.module.refinement.parameters() if args.mGPU else net.refinement.parameters()
        Heatmap_criterion = nn.MSELoss()
        Orientation_criterion = nn.CrossEntropyLoss(train_set.pose_distribution.float().cuda())

        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        timer = Chronometer()
        start_epoch = args.start_epoch

        if not os.path.isdir(args.ckpt):
            os.mkdir(args.ckpt)

        if args.resume:
            checkpoint = torch.load(args.resumed_ckpt)
            start_epoch = checkpoint['epoch']

        best_error, best_accuracy = 1000, 0
        timer.set()
        # Initiate Logger
        with open(args.ckpt + '/logger.txt', 'w+') as f:
            f.write('Training Session on ' + time.strftime("%Y%m%d-%H") + '\n')
            # Write Used Arguments
            f.write('Used Arguments:\n')
            print('Used Arguments:')
            for key in args.__dict__.keys():
                f.write(key + ':{}\n'.format(args.__dict__[key]))
                print(key + ':{}'.format(args.__dict__[key]))

            # Training Loop
            for epoch in range(start_epoch, args.epochs):
                f.write('Epoch: {}'.format(epoch + 1))

                epoch_train_heatmap_loss, epoch_train_orientation_loss, is_best_orinetation, is_best_kp = \
                    0.0, 0.0, False, False

                with tqdm(total=len(train_loader), ncols=0, file=sys.stdout,
                          desc='Epoch: {}'.format(epoch + 1)) as pbar:

                    for i, in_batch in enumerate(train_loader):
                        optimizer.zero_grad()

                        image_in1, image_in2, gt_heatmaps, _, gt_orientation_label = in_batch

                        if torch.cuda.is_available():
                            image_in1, image_in2, gt_heatmaps, gt_orientation_label = image_in1.cuda(),\
                                                                                      image_in2.cuda(),\
                                                                                      gt_heatmaps.cuda(),\
                                                                                      gt_orientation_label.cuda()

                        coarse_kp, fine_kp, orientation = net(image_in1, image_in2)
                        heatmap_loss = Heatmap_criterion(fine_kp, gt_heatmaps)
                        orientation_loss = Orientation_criterion(orientation, gt_orientation_label.squeeze())
                        loss = heatmap_loss + args.Lambda * orientation_loss

                        epoch_train_heatmap_loss += heatmap_loss.item()
                        epoch_train_orientation_loss += orientation_loss.item()
                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix(fine_kp_loss=heatmap_loss.item(), orientation_loss=orientation_loss.item())
                        pbar.update()

                epoch_train_heatmap_loss, epoch_train_orientation_loss = \
                    epoch_train_heatmap_loss / len(train_loader), epoch_train_orientation_loss / len(train_loader)

                # save the checkpoint
                save_checkpoint(
                    {'epoch': epoch + 1, 'net_state_dict': net.module.state_dict() if args.mGPU else net.state_dict()},
                    is_best=False, filename=os.path.join(args.ckpt, 'checkpoint.pth.tar'),
                    best_filename=os.path.join(args.ckpt, 'best_checkpoint.pth.tar'))

                f.write('Average Heatmap & Orientation Loss : {} and {}'.
                      format(epoch_train_heatmap_loss, epoch_train_orientation_loss))

                print('Average Heatmap & Orientation Loss : {} and {}'.
                      format(epoch_train_heatmap_loss, epoch_train_orientation_loss))

                # Check Error of the Trained Model on test set
                if epoch % args.test_every_n_epoch == args.test_every_n_epoch - 1:
                    print('Testing the network...')
                    net.eval()
                    output = test.test(args, net, epoch)
                    print(output['message'])
                    if output['fine_error'] < best_error:
                        best_error = output['fine_error']
                        is_best_kp = True
                    if output['orientation_accuracy'] > best_accuracy:
                        best_accuracy = output['orientation_accuracy']
                        is_best_orinetation = True

                    # save the checkpoint as best checkpoint so far
                    save_checkpoint(
                        {'epoch': epoch + 1,
                         'net_state_dict': net.module.state_dict() if args.mGPU else net.state_dict()},
                        is_best_kp, filename=os.path.join(args.ckpt, 'checkpoint.pth.tar'),
                        best_filename=os.path.join(args.ckpt, 'best_fine_kp_checkpoint.pth.tar'))
                    save_checkpoint(
                        {'epoch': epoch + 1,
                         'net_state_dict': net.module.state_dict() if args.mGPU else net.state_dict()},
                        is_best_orinetation, filename=os.path.join(args.ckpt, 'checkpoint.pth.tar'),
                        best_filename=os.path.join(args.ckpt, 'best_orientation_checkpoint.pth.tar'))

                    f.write('\n')
                    f.write(output['message'])
                    f.write('\n')
                    net.train()

            timer.stop()
            f.write('Finished training session after {0} epochs, {1} hours & {2} minutes, best fine error: '
                    '{3:.2f} pixels in 56 by 56 grid, best orientation accuracy: {4:.2f}%.'
                    .format(args.epochs - start_epoch, int(timer.elapsed / 3600), int((timer.elapsed % 3600) / 60),
                            best_error, best_accuracy))
            f.close()

        print('Finished training session after {0} epochs, {1} hours & {2} minutes, best fine error: '
              '{3:.2f} pixels in 56 by 56 grid, best orientation accuracy: {4:.2f}%.'
              .format(args.epochs - start_epoch, int(timer.elapsed / 3600), int((timer.elapsed % 3600) / 60),
                      best_error, best_accuracy))

