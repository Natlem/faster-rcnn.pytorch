import sys

sys.path.insert(0, "./frcnn")

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient

from model.da_faster_rcnn.vgg16 import vgg16DA
from model.da_faster_rcnn.resnet import resnetDA
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

from frcnn_utils_ga import FasterRCNN_prepare_ga, collect_frcnn_ga, train_frcnn_ga
from frcnn_utils import LoggerForSacred, eval_frcnn, save_state_dict
from visdom_logger.logger import VisdomLogger

import functools
from collections import OrderedDict


def train_eval_fasterRCNN_ga(alpha, epochs, **kwargs):

    frcnn_extra = kwargs["frcnn_extra"]

    optimizer = kwargs["optimizer"]
    model = kwargs["model"]
    device = kwargs["device"]

    logger = kwargs["logger"]
    logger_id = kwargs["logger_id"]

    is_debug = False
    if "is_break" in kwargs:
        is_debug = kwargs["is_debug"]

    kwargs["train_loader"] = frcnn_extra.s_dataloader_train


    best_t_map = -1.
    best_t_ep = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=frcnn_extra.lr_decay_step,
                                          gamma=frcnn_extra.lr_decay_gamma)

    collect_frcnn_ga(alpha, frcnn_extra, device, model, optimizer, is_debug=is_debug)
    g_p = {}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            g_p[n] = p.grad

    for epoch in range(1, epochs + 1):
        train_frcnn_ga(alpha, frcnn_extra, device, model, optimizer, g_p, is_debug=False)


        torch.cuda.empty_cache()

    return best_t_map, best_t_ep

def main():
    lr = 0.001
    momentum = 0.9
    device = torch.device("cuda")
    epochs = 20
    alpha = 0.1
    #source_model_pretrained = 'frcnn_model_vgg16_2_0.7956096921947304_hollywood'
    #source_model_pretrained = 'frcnn_model_vgg16_9_0.8100253828043802_scuta'
    pth_path = "all_saves/frcnn_pth_resnet101_17_0.7665382823348219_scuta_head"
    #pth_path = "all_saves/frcnn_pth_vgg16_1_0.7456834728436339_hollywood_head"

    # Model Config
    net = "resnet101"
    pretrained = True

    batch_size = 1
    frcnn_extra = FasterRCNN_prepare_ga(net, batch_size, "scuta", "scutb", cfg_file="cfgs/vgg16.yml", debug=True)

    frcnn_extra.ga_forward()

    if frcnn_extra.net == "vgg16":
        fasterRCNN = vgg16DA(frcnn_extra.s_imdb_train.classes, pretrained=pretrained, class_agnostic=frcnn_extra.class_agnostic, model_path="", pth_path="data/pretrained_model/vgg16_caffe.pth")
    if frcnn_extra.net == "resnet101":
        lr = 0.001
        epochs = 40
        frcnn_extra.lr_decay_step = 10
        fasterRCNN = resnetDA(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                            class_agnostic=frcnn_extra.class_agnostic, model_path="",
                            pth_path="")
    fasterRCNN.create_architecture()
    if pth_path != "":
        ck = torch.load(pth_path)
        fasterRCNN.load_state_dict(ck['model'], strict=False)
        cfg.POOLING_MODE = ck['pooling_mode']
    #fasterRCNN = torch.load(source_model_pretrained)

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
    #src_map = eval_frcnn(frcnn_extra, device, fasterRCNN, False)
    #tar_map = eval_frcnn_da(frcnn_extra, device, fasterRCNN, False)
    train_eval_fasterRCNN_ga(alpha, epochs, device=device, model=fasterRCNN, optimizer=optimizer,
                   logger=logger, logger_id=1, frcnn_extra=frcnn_extra, is_debug=True)



if __name__ == "__main__":
    main()
