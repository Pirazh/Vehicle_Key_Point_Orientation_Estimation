import warnings
from models import KP_Orientation_Net
import torch
from tools import train, test
import argparse
import time


def main(args):

    warnings.filterwarnings('ignore')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.use_case == 'stage1':
        net = KP_Orientation_Net.CoarseRegressor()
    elif args.use_case == 'stage2':
        net = KP_Orientation_Net.KeyPointModel()
        if args.phase == 'train':
            # Load the stage 1 checkpoint and freeze its weights
            net.coarse_estimator.load_state_dict(torch.load(args.stage1_ckpt)['net_state_dict'])
            for param in net.coarse_estimator.parameters():
                param.requires_grad = False
            print('stage1 weights have been initialized with pre-trained weights and are frozen!')
    else:
        raise NameError('use case should be either "stage1" or "stage2"')

    net = net.to(device)

    print('Total number of Parameters = %s' % sum(p.numel() for p in net.parameters()))
    print('Total number of trainable Parameters = %s' % sum(p.numel() for p in net.parameters() if p.requires_grad))

    if args.resume or args.phase == 'test':
        checkpoint = torch.load(args.resumed_ckpt)
        net.load_state_dict(checkpoint['net_state_dict'])
        print('Resumed Checkpoint :{} is Loaded!'.format(args.resumed_ckpt))

    if torch.cuda.device_count() > 1 and args.mGPU:
        net = torch.nn.DataParallel(net)

    if args.phase == 'train':
        train.train(args, net)
    elif args.phase == 'test':
        net.eval()
        output = test.test(args, net)
        print(output['message'])
    else:
        raise NameError('phase should be either "train" or "test"')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Key-Point and Orientation Estimation Network')
    parser.add_argument('--phase', default='train', type=str, choices=['train', 'test'],
                        help='train/test mode selection', required=True)
    parser.add_argument('--use_case', default='Stage1', type=str, choices=['stage1', 'stage2'],
                        help='Coarse/Fine heatmap model training', required=True)
    parser.add_argument('--rotate_probability', default=0, type=float)
    parser.add_argument('--flip_probability', default=0, type=float)
    parser.add_argument('--visualize', default=False, action='store_true',
                        help='randomly visulaize estimated key-points with respect to GT')
    parser.add_argument('--train_batch_size', default=128, help='Size of Training Batch', type=int)
    parser.add_argument('--test_batch_size', default=128, help='Size of Testing Batch', type=int)
    parser.add_argument('--lr', default=0.0001, help='Learning Rate', type=float)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--weight_decay', default=0, help='Optimizer Weight Decay', type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--Lambda', default=0.1, type=float, help='The balance between fine/coarse losses')
    parser.add_argument('--test_every_n_epoch', default=1, help='Testing Network after n epochs', type=int)
    parser.add_argument('--resume', default=False, action='store_true', help='resume to specific checkpoint')
    parser.add_argument('--mGPU', default=False, action='store_true', help='Multi GPU support')
    parser.add_argument('--stage1_ckpt', default='', help='Path to the stage1 trained model', type=str,
                        required=True if (parser.parse_known_args()[0].use_case == 'stage2' and
                                         parser.parse_known_args()[0].phase == 'train') else False)
    parser.add_argument('--resumed_ckpt', default='', help='Path to resume the checkpoint', type=str,
                        required=True if parser.parse_known_args()[0].phase == 'test' or
                                         parser.parse_known_args()[0].resume else False)
    if parser.parse_known_args()[0].phase == 'train':
        parser.add_argument('--ckpt',
                            default='./checkpoints/' + parser.parse_known_args()[0].use_case +
                                    '/' + time.strftime("%Y-%m-%d-%H"),
                            help='Path to save the checkpoints',
                            type=str)
    args = parser.parse_args()
    main(args)
