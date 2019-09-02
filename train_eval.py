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

from frcnn_utils import FasterRCNN_prepare, train_frcnn, eval_frcnn, LoggerForSacred, save_state_dict
from visdom_logger.logger import VisdomLogger

import functools
from collections import OrderedDict

def train_eval_fasterRCNN(epochs,  **kwargs):

    frcnn_extra = kwargs["frcnn_extra"]
    optimizer = kwargs["optimizer"]
    model = kwargs["model"]
    device = kwargs["device"]
    logger = kwargs["logger"]
    logger_id = kwargs["logger_id"]

    is_debug = False
    if "is_debug" in kwargs:
        is_debug = kwargs["is_debug"]

    best_map = -1.
    best_ep = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=frcnn_extra.lr_decay_step, gamma=frcnn_extra.lr_decay_gamma)
    for epoch in range(1, epochs + 1):

        total_loss = train_frcnn(frcnn_extra, device, model, optimizer, is_debug)
        scheduler.step()

        map = eval_frcnn(frcnn_extra, device, model, is_debug)

        if not is_debug:
            if map > best_map:
                best_map = map
                best_ep = epoch
            torch.save(model, "{}/frcnn_model_{}_{}_{}_{}".format("all_saves", frcnn_extra.net, epoch, map, frcnn_extra.s_dataset))
            torch.save(optimizer, "{}/frcnn_op_model_{}_{}_{}_{}".format("all_saves",frcnn_extra.net, epoch, map, frcnn_extra.s_dataset))
            save_state_dict(model, optimizer, frcnn_extra.class_agnostic, "{}/frcnn_pth_{}_{}_{}_{}_head".format("all_saves",frcnn_extra.net, epoch, map, frcnn_extra.s_dataset))
        if logger is not None:
            logger.log_scalar("frcnn_{}_{}_training_loss".format(frcnn_extra.net, logger_id), total_loss, epoch)
            logger.log_scalar("frcnn_{}_{}_target_val_acc".format(frcnn_extra.net, logger_id), map, epoch)
        torch.cuda.empty_cache()

    return best_ep, best_map

def main():

    momentum = 0.9
    device = torch.device("cuda")

    # Model Config
    net = "resnet101"
    pretrained = True

    batch_size = 1
    frcnn_extra = FasterRCNN_prepare(net, batch_size, "pascal_voc_person", cfg_file="cfgs/{}.yml".format(net))
    frcnn_extra.forward()


    if frcnn_extra.s_dataset == 'scuta' or frcnn_extra.s_dataset == 'scuta_ori':
        if frcnn_extra.net == "vgg16":
            lr = 0.01
            epochs = 20
            fasterRCNN = vgg16(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               pth_path='data/pretrained_model/{}_caffe.pth'.format(net))
        if frcnn_extra.net == "resnet101":
            lr = 0.001
            epochs = 40
            frcnn_extra.lr_decay_step = 10
            fasterRCNN = resnet(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               pth_path='data/pretrained_model/{}_caffe.pth'.format(net))
    elif frcnn_extra.s_dataset == 'hollywood':
        if frcnn_extra.net == "vgg16":
            lr = 0.01
            epochs = 20
            fasterRCNN = vgg16(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               pth_path='data/pretrained_model/{}_caffe.pth'.format(net))
        if frcnn_extra.net == "resnet101":
            lr = 0.001
            epochs = 40
            frcnn_extra.lr_decay_step = 10
            fasterRCNN = resnet(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               pth_path='data/pretrained_model/{}_caffe.pth'.format(net))
    elif frcnn_extra.s_dataset == 'pascal_voc_person':
        if frcnn_extra.net == "vgg16":
            lr = 0.01
            epochs = 20
            fasterRCNN = vgg16(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               pth_path='data/pretrained_model/{}_caffe.pth'.format(net))
        if frcnn_extra.net == "resnet101":
            lr = 0.001
            epochs = 10
            frcnn_extra.lr_decay_step = 10
            fasterRCNN = resnet(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               pth_path='data/pretrained_model/{}_caffe.pth'.format(net))

    elif frcnn_extra.s_dataset == 'pascal_voc':
        if frcnn_extra.net == "vgg16":
            lr = 0.01
            epochs = 20
            fasterRCNN = vgg16(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               pth_path='data/pretrained_model/{}_caffe.pth'.format(net))
        if frcnn_extra.net == "resnet101":
            lr = 0.001
            epochs = 10
            frcnn_extra.lr_decay_step = 10
            fasterRCNN = resnet(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                               class_agnostic=frcnn_extra.class_agnostic,
                               pth_path='data/pretrained_model/{}_caffe.pth'.format(net))


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

    logger = VisdomLogger(port=10999)
    logger = LoggerForSacred(logger)

    train_eval_fasterRCNN(3, frcnn_extra=frcnn_extra, optimizer=optimizer, model=fasterRCNN,
                          device=device, logger=logger, logger_id="-1", is_debug=True)




if __name__ == "__main__":
    main()
