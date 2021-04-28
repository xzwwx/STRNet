import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something'], default="something")
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'], default="RGB")
parser.add_argument('--train_list', type=str, default="/data/Disk_A/zhiwei/SomethingCode/train_videofolder.txt")#/data/Disk_A/zhiwei/Code/STRNet/dataset/ucf101/ucf101_30fps_rgb_train_split_1.txt")#"/data/Disk_A/zhiwei/SomethingCode/train_videofolder.txt")#
parser.add_argument('--val_list', type=str, default="/data/Disk_A/zhiwei/SomethingCode/val_videofolder.txt")#/data/Disk_A/zhiwei/Code/STRNet/dataset/ucf101/ucf101_30fps_rgb_val_split_1.txt")#"/data/Disk_A/zhiwei/SomethingCode/val_videofolder.txt")#

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="STR")
parser.add_argument('--num_frames', type=int, default=16)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 30], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip_gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

parser.add_argument('--pretrained_parts', type=str, default='scratch',
                    choices=['scratch', '2D', '3D', 'both', 'finetune'])
parser.add_argument('--net_modelECO', type=str, default="")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)








