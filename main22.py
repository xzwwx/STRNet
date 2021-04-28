import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from torch.nn.init import xavier_uniform_, constant_

from dataset import STRDataSet
from STRNet_model import STRNet
from transforms import *
from opts import parser
from tensorboardX import SummaryWriter

best_prec1 = 0



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'something':
        num_class = 174
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = STRNet(num_classes=num_class, num_frames=args.num_frames, pretrained_parts=args.pretrained_parts)

    crop_size = None                # model.crop_size
    scale_size = None               # model.scale_size
    input_mean = [0.485,0.456,0.406]               # model.input_mean  [0.485,0.456,0.406]
    input_std = [0.229,0.224,0.225]                # model.input_std   [0.229,0.224,0.225]
    policies = None                 # model.get_optim_policies()

    train_augmentation = model.get_augmentation(mode='train')       # model.get_augmentation()
    val_aug = model.get_augmentation(mode='val')
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    model_dict = model.state_dict()
    # print(model_dict)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    else:
        if args.arch == "ECO":

            if args.pretrained_parts == "scratch":
                new_state_dict = {}

            elif args.pretrained_parts == "finetune":
                print(args.net_modelECO)
                print("66" * 40)
                if args.net_modelECO is not None:
                    pretrained_dict = torch.load(args.net_modelECO)
                    #kinetics_pretrained_dict = torch.load("/data/Disk_A/zhiwei/Code/XzwModel/Kinetics_40_223.pth.tar")
                    print(("=> loading model-finetune: '{}'".format(args.net_modelECO)))
                    # print(kinetics_pretrained_dict['state_dict'].keys())
                else:
                    pretrained_dict = torch.load("models/eco_lite_rgb_16F_kinetics_v2.pth.tar")
                    print(("=> loading model-finetune-url: '{}'".format("models/eco_lite_rgb_16F_kinetics_v2.pth.tar")))

                # calibrate resnet odel
                # temp_dict = {k : v for k,v in pretrained_dict.items() if k in (for l in model_dict.keys())}
                '''
                temp_dict = {}
                for k,v in pretrained_dict.items():
                    for l in model_dict.keys():
                        if k in l and (v.size() == model_dict[l].size()):
                            temp_dict[l] = v
                            print(l)
                            break
                '''
                '''
                for k,v in kinetics_pretrained_dict['state_dict'].items():
                    for l in model_dict.keys():
                        if k in l and (v.size() == model_dict[l].size()) and 'res' in l :
                            temp_dict[l] = v
                            print(l)
                            break
                '''
                new_state_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if
                              (k in model_dict) and (v.size() == model_dict[k].size())}
                #new_state_dict = temp_dict
                print("*" * 50)
                print("Start finetuning ..")

        else:
            new_state_dict ={}
            #temp_dict = {}

        un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
        # print("un_init_dict_keys: ", un_init_dict_keys)
        print("\n------------------------------------")

        for k in un_init_dict_keys:
            new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
            if 'weight' in k:
                if 'bn' in k:
                    print("{} init as: 1".format(k))
                    constant_(new_state_dict[k], 1)
                else:
                    print("{} init as: xavier".format(k))
                    xavier_uniform_(new_state_dict[k])
            elif 'bias' in k:
                print("{} init as: 0".format(k))
                constant_(new_state_dict[k], 0)

        print("------------------------------------")
        model.load_state_dict(new_state_dict)

    cudnn.benchmark = True

    normalize = GroupNormalize(input_mean, input_std)

    # Load data
    train_loader = None
    val_loader = None
    train_loader = torch.utils.data.DataLoader(
    #STRDataSet('/data/Disk_A/zhiwei/Data', args.train_list,
        STRDataSet('/data/Disk_C/something/20bn-something-something-v1', args.train_list,
                   sample_frames=args.num_frames,
                   modality=args.modality,
                   image_tmpl="{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(),
                       ToTorchFormatTensor(),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,drop_last=True)

    val_loader = torch.utils.data.DataLoader(
    #STRDataSet('/data/Disk_A/zhiwei/Data', args.val_list,
        STRDataSet('/data/Disk_C/something/20bn-something-something-v1', args.val_list,
                   sample_frames=args.num_frames,
                   modality=args.modality,
                   image_tmpl="{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       val_aug,
                       Stack(),
                       ToTorchFormatTensor(),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,drop_last=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total:', total_num/1000000, 'M. Trainable: ', trainable_num/1000000, 'M.')



    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, 0.1)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    writer = SummaryWriter('logs/{}'.format(log_time))


    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        losses0, top10, top50, lr0 = train(train_loader, model, criterion, optimizer, epoch)
        writer.add_scalar('train/loss', losses0, epoch)
        writer.add_scalar('train/top1', top10, epoch)
        writer.add_scalar('train/top5', top50, epoch)
        writer.add_scalar('train/lr', lr0, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, prec5, loss0 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
            writer.add_scalar('eval/top1', prec1, epoch)
            writer.add_scalar('eval/top5', prec5, epoch)
            writer.add_scalar('eval/loss', loss0, epoch)
        scheduler.step()
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # if args.no_partialbn:
    #     model.module.partialBN(False)
    # else:
    #     model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        # print(input_var.size())
        input_var = input_var.view((-1, 3) + input_var.size()[-2:])
        # print(input_var.size())
        output = model(input_var)
        out1 = output[0]
        out2 = output[1]
        out3 = output[2]
        out4 = output[3]

        out = out1 #+0.5 * out2 + 0.8 * out3  + 0.5 * out4
        # print(out.size())
        loss = criterion(out, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

    return losses.avg, top1.avg, top5.avg, optimizer.param_groups[-1]['lr']


def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    begin_time = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        input_var = input_var.view((-1, 3) + input_var.size()[-2:])

        # compute output
        with torch.no_grad():
            output = model(input_var)[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5)))

    end_time = time.time()
    print("FPS:", ((i * args.batch_size)/(end_time - begin_time)) * 16)
    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == '__main__':
    main()

