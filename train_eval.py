import sys

sys.path.insert(0, "./frcnn")

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
import numpy as np
from model.roi_layers import nms

from torch.utils.data.sampler import Sampler
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time

from frcnn_utils import FasterRCNN_prepare, train_frcnn, eval_frcnn, LoggerForSacred
from visdom_logger.logger import VisdomLogger

import functools
from collections import OrderedDict



def train_eval_fasterRCNN(epochs, **kwargs):

    frcnn_extra = kwargs["frcnn_extra"]

    optimizer = kwargs["optimizer"]
    model = kwargs["model"]
    cuda = kwargs["cuda"]

    logger = kwargs["logger"]
    logger_id = frcnn_extra.dataset

    is_break = False
    if "is_break" in kwargs and kwargs["is_break"]:
        is_break = True

    kwargs["train_loader"] = frcnn_extra.dataloader_train

    loss_acc = []
    lr = optimizer.param_groups[0]['lr']

    for epoch in range(1, epochs + 1):

        total_loss = train_frcnn(frcnn_extra, cuda, model, optimizer, is_break)

        if epoch % (frcnn_extra.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, frcnn_extra.lr_decay_gamma)
            lr *= frcnn_extra.lr_decay_gamma

        map = eval_frcnn(frcnn_extra, cuda, model, is_break)
        torch.save(model, "frcnn_model_{}_{}_{}_{}".format(frcnn_extra.net, epoch, map, frcnn_extra.dataset))
        torch.save(optimizer, "frcnn_op_model_{}_{}_{}_{}".format(frcnn_extra.net, epoch, map, frcnn_extra.dataset))
        if logger is not None:
            logger.log_scalar("frcnn_{}_{}_training_loss".format(frcnn_extra.net, logger_id), total_loss, epoch)
            logger.log_scalar("frcnn_{}_{}_target_val_acc".format(frcnn_extra.net, logger_id), map, epoch)
        torch.cuda.empty_cache()


    return loss_acc

def main():

    momentum = 0.9
    device = torch.device("cuda")

    # Model Config
    net = "resnet101"
    pretrained = True

    batch_size = 1
    frcnn_extra = FasterRCNN_prepare(net, batch_size, "hollywood", "cfgs/{}.yml".format(net))
    frcnn_extra.forward()

    if frcnn_extra.dataset == 'scuta':
        if frcnn_extra.net == "vgg16":
            lr = 0.01
            epochs = 20
            fasterRCNN = vgg16(frcnn_extra.imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               model_path='data/pretrained_model/{}_caffe.pth'.format(net))
        if frcnn_extra.net == "resnet101":
            lr = 0.001
            epochs = 40
            frcnn_extra.lr_decay_step = 10
            fasterRCNN = resnet(frcnn_extra.imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               model_path='data/pretrained_model/{}_caffe.pth'.format(net))
    elif frcnn_extra.dataset == 'hollywood':
        if frcnn_extra.net == "vgg16":
            lr = 0.01
            epochs = 20
            fasterRCNN = vgg16(frcnn_extra.imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               model_path='data/pretrained_model/{}_caffe.pth'.format(net))
        if frcnn_extra.net == "resnet101":
            lr = 0.001
            epochs = 40
            frcnn_extra.lr_decay_step = 10
            fasterRCNN = resnet(frcnn_extra.imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               model_path='data/pretrained_model/{}_caffe.pth'.format(net))

    fasterRCNN.create_architecture()
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    optimizer = torch.optim.SGD(params, momentum=momentum, lr=lr)
    fasterRCNN = fasterRCNN.to(device)

    logger = VisdomLogger(port=9000)
    logger = LoggerForSacred(logger)

    train_eval_fasterRCNN(epochs, cuda=device, model=fasterRCNN, optimizer=optimizer,
                   logger=logger, frcnn_extra=frcnn_extra, is_break=False)



if __name__ == "__main__":
    main()
