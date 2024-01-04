from __future__ import print_function

import os
import sys
import argparse
import time
import math

#import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier
'''
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
'''

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.6,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,70,80',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'KD_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def teacher_set_model(opt):
    model = SupConResNet(name='resnet34')
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name='resnet34', num_classes=opt.n_cls)

    ckpt = torch.load('save/SupCon/cifar10_models/SupCon_cifar10_resnet34_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_0/last.pth')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion


def student_set_model(opt):
    model = SupConResNet(name='resnet18')
    criterion = torch.nn.CrossEntropyLoss()
    '''
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)
    '''
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train_knowledge_distillation(teacher_model, student_model, train_loader, epoch, optimizer, opt):
    """one epoch training"""
    teacher_model.eval()  # Teacher set to evaluation mode
    student_model.train() # Student to train mode

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    criterion = torch.nn.MSELoss()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        with torch.no_grad():
            teacher_features = teacher_model.encoder(images) 

        # Forward pass with the student model
        student_features = student_model.encoder(images)

        loss = criterion(teacher_features, student_features)
        
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


opt = parse_option()

# build data loader
train_loader, val_loader = set_loader(opt)

# build model and criterion
teacher_model, classifier, criterion = teacher_set_model(opt)
student_model, criterion = student_set_model(opt)
# build optimizer
optimizer_s = set_optimizer(opt, student_model)

# tensorboard
# logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
tensorboard_writer = SummaryWriter(log_dir=os.path.join(opt.save_folder, 'tensorboard_logs'))

# training routine
for epoch in range(1, opt.epochs + 1):
    adjust_learning_rate(opt, optimizer_s, epoch)

    # train for one epoch
    time1 = time.time()
    loss = train_knowledge_distillation(teacher_model, student_model, train_loader, epoch, optimizer_s, opt)
    time2 = time.time()
    print('student model Train epoch {}, total time {:.2f}'.format(
        epoch, time2 - time1))

    #tensorboard logger
    #logger.log_value('loss', loss, epoch)
    #logger.log_value('learning_rate', optimizer_s.param_groups[0]['lr'], epoch)
    tensorboard_writer.add_scalar('Train/Loss', loss, epoch)

    if epoch % opt.save_freq == 0:
        save_file = os.path.join(
            opt.save_folder, 'ckpt_epoch_{epoch}_KD.pth'.format(epoch=epoch))
        save_model(student_model, optimizer_s, opt, epoch, save_file)

    # save the last model
save_file = os.path.join(
    opt.save_folder, 'last.pth')
save_model(student_model, optimizer_s, opt, opt.epochs, save_file)

tensorboard_writer.close()