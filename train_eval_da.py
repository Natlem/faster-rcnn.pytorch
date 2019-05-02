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
from model.faster_rcnn.domain_adapt import D_cls_image, D_cls_inst, consistency_reg

from torch.utils.data.sampler import Sampler
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from frcnn_utils import FasterRCNN_prepare_da, train_frcnn, train_frcnn_da, eval_frcnn, eval_frcnn_da, LoggerForSacred
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
    d_cls_image = D_cls_image().to(cuda)
    d_cls_inst = D_cls_inst().to(cuda)
    d_image_opt = torch.optim.SGD(d_cls_image.parameters(), lr=lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                                  weight_decay=cfg.TRAIN.WEIGHT_DECAY, momentum=cfg.TRAIN.MOMENTUM)
    d_inst_opt = torch.optim.SGD(d_cls_inst.parameters(), lr=lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY, momentum=cfg.TRAIN.MOMENTUM)
    d_cst_src_loss = 0
    d_cst_tar_loss = 0

    for epoch in range(1, epochs + 1):
        start_steps = epoch * len(frcnn_extra.dataloader_train)
        total_steps = epochs * len(frcnn_extra.dataloader_train)
        total_loss = train_frcnn_da(frcnn_extra, cuda, model, optimizer, d_cls_image, d_cls_inst, d_image_opt, d_inst_opt,
                                    start_steps, total_steps, is_break)

        if epoch % (frcnn_extra.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, frcnn_extra.lr_decay_gamma)
            lr *= frcnn_extra.lr_decay_gamma

        src_map = eval_frcnn(frcnn_extra, cuda, model, is_break)
        tar_map = eval_frcnn_da(frcnn_extra, cuda, model, is_break)
        torch.save(model, "frcnn_da_model_{}_{}_{}_{}".format(frcnn_extra.net, epoch, tar_map, frcnn_extra.dataset))
        torch.save(optimizer, "frcnn_da_op_model_{}_{}_{}_{}".format(frcnn_extra.net, epoch, tar_map, frcnn_extra.dataset))
        if logger is not None:
            logger.log_scalar("frcnn_da_{}_{}_training_loss".format(frcnn_extra.net, logger_id), total_loss, epoch)
            logger.log_scalar("frcnn_da_{}_{}_src_val_acc".format(frcnn_extra.net,logger_id), src_map, epoch)
            logger.log_scalar("frcnn_da_{}_{}_tar_val_acc".format(frcnn_extra.net, logger_id), tar_map, epoch)
        torch.cuda.empty_cache()


    return loss_acc

def main():
    lr = 0.001
    momentum = 0.9
    device = torch.device("cuda")
    epochs = 20

    # Model Config
    net = "vgg16"
    pretrained = True

    batch_size = 1
    frcnn_extra = FasterRCNN_prepare_da(net, batch_size, "holly", "src" "cfgs/vgg16.yml")
    frcnn_extra.forward()

    if frcnn_extra.net == "vgg16":
        fasterRCNN = vgg16(frcnn_extra.imdb_train.classes, pretrained=pretrained, class_agnostic=frcnn_extra.class_agnostic, model_path='data/pretrained_model/vgg16_caffe.pth')

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
