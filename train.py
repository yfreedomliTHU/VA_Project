#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
from optparse import OptionParser
from tools.config_tools import Config

#----------------------------------- loading paramters -------------------------------------------#
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/train_config.yaml")
(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
print(opt)

parser_test = OptionParser()
parser_test.add_option('--config',
                    type=str,
                    help="evaluation configuration",
                    default="./configs/test_config.yaml")

(opts_test, args_test) = parser_test.parse_args()
assert isinstance(opts_test, object)
opt_test = Config(opts_test.config)
print(opt_test)
#--------------------------------------------------------------------------------------------------#

#------------------ environment variable should be set before import torch  -----------------------#
if opt.cuda:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    print('setting gpu on gpuid {0}'.format(opt.gpu_id))
#--------------------------------------------------------------------------------------------------#

import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

import models
from dataset import VideoFeatDataset as dset
from tools import utils
#from logger import Logger


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")

# setting the random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
if opt.cuda and torch.cuda.is_available():
    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.manualSeed)
else:
    torch.manual_seed(opt.manualSeed)
print('Random Seed: {0}'.format(opt.manualSeed))

# make checkpoint folder
if opt.checkpoint_folder is None:
    opt.checkpoint_folder = 'checkpoints'
if not os.path.exists(opt.checkpoint_folder):
    os.system('mkdir {0}'.format(opt.checkpoint_folder))

# loading dataset
train_dataset = dset(opt.data_dir, flist=opt.flist)
print('number of train samples is: {0}'.format(len(train_dataset)))
print('finished loading data')

max_ans = 0

def test(video_loader, audio_loader, model, opt):
    """
    train for one epoch on the training set
    """
    # training mode
    model.eval()

    global max_ans

    right = 0
    for _, vfeat in enumerate(video_loader):
        for _, afeat in enumerate(audio_loader):
            # shuffling the index orders
            bz = vfeat.size()[0]
            for k in np.arange(bz):
                cur_vfeat = vfeat[k].clone()
                cur_vfeats = cur_vfeat.repeat(bz, 1, 1)

                vfeat_var = Variable(cur_vfeats)
                afeat_var = Variable(afeat)

                if opt.cuda:
                    vfeat_var = vfeat_var.cuda()
                    afeat_var = afeat_var.cuda()

                v,a = model(vfeat_var, afeat_var)
                cur_sim = torch.abs(v-a)
                cur_sim = torch.sum(torch.pow(cur_sim,2),1)
                cur_sim = torch.sqrt(cur_sim)
                cur_sim = cur_sim.view(bz,1)
                
                if k == 0:
                    simmat = cur_sim.clone()
                else:
                    simmat = torch.cat((simmat, cur_sim), 1)
            sorted, indices = torch.sort(simmat, 0)
            np_indices = indices.cpu().data.numpy()
            topk = np_indices[:opt.topk,:]
            for k in np.arange(bz):
                order = topk[:,k]
                if k in order:
                    right = right + 1
            print('Testing accuracy (top{}): {:.3f}'.format(opt.topk, right/bz))
            if right/bz > max_ans:
                max_ans = right/bz
                path_checkpoint = '{0}/{1}_state_Max_epoch.pth'.format(opt.checkpoint_folder, opt.model)
                utils.save_checkpoint(model, path_checkpoint)

def to_np(x):
    return x.data.cpu().numpy()
# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    #logger = Logger('./logs')

    # training mode
    model.train()

    end = time.time()
    for i, (vfeat, afeat) in enumerate(train_loader):
        # shuffling the index orders
        bz = vfeat.size()[0]
        orders = np.arange(bz).astype('int32')
        shuffle_orders = orders.copy()
        
        np.random.shuffle(shuffle_orders)
        while sum(shuffle_orders == orders) > 0:
            shuffle_orders = orders.copy()
            np.random.shuffle(shuffle_orders)

        # creating a new data with the shuffled indices
        afeat2 = afeat[torch.from_numpy(shuffle_orders).long()].clone()

        # put the data into Variable
        vfeat_var = Variable(vfeat)
        afeat_varp = Variable(afeat)
        afeat_varn = Variable(afeat2)
        #target_var = Variable(target)

        # if you have gpu, then shift data to GPU
        if opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_varp = afeat_varp.cuda()
            afeat_varn = afeat_varn.cuda()
            #target_var = target_var.cuda()

        # forward, backward optimize
        v1,ap = model(vfeat_var,afeat_varp)
        v2,an = model(vfeat_var,afeat_varn)
        anc = (v1+v2)/2
        posi = ap
        nega = an
        loss = criterion(anc,posi,nega)


        ##############################
        # update loss in the loss meter
        ##############################
        losses.update(loss.data[0], vfeat.size(0))

        ##############################
        # compute gradient and do sgd
        ##############################
        optimizer.zero_grad()
        loss.backward()

        ##############################
        # gradient clip stuff
        ##############################
        #utils.clip_gradient(optimizer, opt.gradient_clip)

        # update parameters
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses)
            print(log_str)

# learning rate adjustment function
def LR_Policy(optimizer, init_lr, policy):
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * policy

def main():
    global opt
    # train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))

    # create model
    if opt.model is 'VAMetric':
        model = models.VAMetric()
    elif opt.model is 'VAMetric2':
        model = models.VAMetric2()
    else:
        model = models.VAMetric()
        opt.model = 'VAMetric'

    if opt.init_model != '':
        print('loading pretrained model from {0}'.format(opt.init_model))
        model.load_state_dict(torch.load(opt.init_model))

    # Contrastive Loss
    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.ContrastiveLoss()
    #criterion = torch.nn.CosineEmbeddingLoss(margin=0.2)
    criterion = torch.nn.TripletMarginLoss(margin=1.0)

    if opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), opt.lr,
                                weight_decay=opt.weight_decay)

    # adjust learning rate every lr_decay_epoch
    lambda_lr = lambda epoch: opt.lr_decay ** ((epoch + 1) // opt.lr_decay_epoch)   #poly policy
    
    global opt_test
    test_video_dataset = dset(opt_test.data_dir, opt_test.video_flist, which_feat='vfeat')
    test_audio_dataset = dset(opt_test.data_dir, opt_test.audio_flist, which_feat='afeat')
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=opt_test.batchSize,
                                                    shuffle=False, num_workers=int(opt_test.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=opt_test.batchSize,
                                                    shuffle=False, num_workers=int(opt_test.workers))

    for epoch in range(opt.max_epochs):
    	#################################
        # train for one epoch
        #################################
        train(train_loader, model, criterion, optimizer, epoch, opt)
        LR_Policy(optimizer, opt.lr, lambda_lr(epoch))      # adjust learning rate through poly policy
        test(test_video_loader, test_audio_loader, model, opt_test)
        ##################################
        # save checkpoints
        ##################################

        # save model every 10 epochs
        if ((epoch+1) % opt.epoch_save) == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(opt.checkpoint_folder, opt.model, epoch+1)
            utils.save_checkpoint(model, path_checkpoint)

if __name__ == '__main__':
    main()
