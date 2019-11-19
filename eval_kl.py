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

from frcnn_utils import FasterRCNN_prepare, train_frcnn, eval_frcnn, LoggerForSacred, save_state_dict, save_conf, eval_kl_frcnn
from visdom_logger.logger import VisdomLogger

import functools
from collections import OrderedDict



def train_eval_fasterRCNN(**kwargs):

    frcnn_extra = kwargs["frcnn_extra"]

    model_1 = kwargs["model_1"]
    model_2 = kwargs["model_2"]
    cuda = kwargs["cuda"]

    logger = kwargs["logger"]
    logger_id = frcnn_extra.s_dataset

    is_break = False
    if "is_break" in kwargs and kwargs["is_break"]:
        is_break = True

    kwargs["train_loader"] = frcnn_extra.s_dataloader_train

    loss_acc = []


    kl_feat_img_temp, kl_feat_inst_temp, mmd_feat_temp, mmd_inst_temp = eval_kl_frcnn(frcnn_extra, cuda, model_1, model_2, is_break)
    print(kl_feat_img_temp)
    print(kl_feat_inst_temp)
    print(mmd_feat_temp)
    print(mmd_inst_temp)


    return loss_acc

def main():

    momentum = 0.9
    device = torch.device("cuda")

    # Model Config
    net = "vgg16"
    pretrained = True

    batch_size = 1
    frcnn_extra = FasterRCNN_prepare(net, batch_size, "scutb", "cfgs/{}.yml".format(net))
    frcnn_extra.forward()

    # Upper bound vs DA(100)
    model_1_pth = 'best_models/frcnn_pth_vgg16_20_0.8859604884411493_scutb_head'
    model_1_model = ''
    model_2_pth = ''
    model_2_model = 'best_models/frcnn_da_100_img_model_vgg_16_20_0.62_scutb'

    # Lower Bound(all) vs DA(all)
    model_1_pth = 'best_models/frcnn_pth_vgg16_20_0.8214922261155124_scuta_head'
    model_1_model = ''
    model_2_pth = ''
    model_2_model = 'best_models/frcnn_da_img_model_vgg16_20_0.6363939966469591_scutb'

    # Lower Bound(all) vs UpperBound(all)
    model_1_pth = 'best_models/frcnn_pth_vgg16_20_0.8214922261155124_scuta_head'
    model_1_model = ''
    model_2_pth = 'best_models/frcnn_pth_vgg16_20_0.8859604884411493_scutb_head'
    model_2_model = ''

    # Upper bound vs DA(all )
    model_1_pth = 'best_models/frcnn_pth_vgg16_20_0.8859604884411493_scutb_head'
    model_1_model = ''
    model_2_pth = ''
    model_2_model = 'best_models/frcnn_da_img_model_vgg16_20_0.6363939966469591_scutb'

    model_1 = vgg16(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                       class_agnostic=frcnn_extra.class_agnostic,
                       pth_path='data/pretrained_model/{}_caffe.pth'.format(net))

    model_2 = vgg16(frcnn_extra.s_imdb_train.classes, pretrained=pretrained,
                       class_agnostic=frcnn_extra.class_agnostic,
                       pth_path='data/pretrained_model/{}_caffe.pth'.format(net))

    model_1.create_architecture()
    model_2.create_architecture()
    if model_1_pth:
        checkpoint = torch.load(model_1_pth)
        model_1.load_state_dict(checkpoint['model'])
    else:
        model_1 = torch.load(model_1_model)

    if model_2_pth:
        checkpoint = torch.load(model_2_pth)
        model_2.load_state_dict(checkpoint['model'])
    else:
        model_2 = torch.load(model_2_model)

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)

    logger = VisdomLogger(port=10999)
    logger = LoggerForSacred(logger)

    train_eval_fasterRCNN(cuda=device, model_1=model_1, model_2=model_2,
                   logger=logger, frcnn_extra=frcnn_extra, is_break=False)



if __name__ == "__main__":
    main()
