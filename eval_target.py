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

from frcnn_utils import FasterRCNN_prepare_da, FasterRCNN_prepare, train_frcnn, train_frcnn_da, eval_frcnn, eval_frcnn_da, LoggerForSacred, train_frcnn_da_img
from visdom_logger.logger import VisdomLogger

import functools
from collections import OrderedDict





def eval_fasterRCNN(**kwargs):

    frcnn_extra = kwargs["frcnn_extra"]
    model = kwargs["model"]
    cuda = kwargs["cuda"]

    is_break = False
    if "is_break" in kwargs and kwargs["is_break"]:
        is_break = True


    src_map = eval_frcnn(frcnn_extra, cuda, model, is_break)

    return src_map

def main():

    device = torch.device("cuda")
    pth_pretrained = "all_saves/frcnn_pth_vgg16_17_0.822969253226584_scuta_head"
    source_pretrained = "frcnn_model_vgg16_9_0.8100253828043802_scuta"
    #source_pretrained = "frcnn_model_vgg16_2_0.8605263157894737_hollywood"

    # Model Config
    net = "vgg16"
    pretrained = True

    batch_size = 1
    #frcnn_extra = FasterRCNN_prepare_da(net, batch_size, "hollywood", "scuta", "cfgs/vgg16.yml"
    frcnn_extra = FasterRCNN_prepare(net, batch_size, "scuta", "cfgs/vgg16.yml")

    frcnn_extra.forward(is_training=True)

    if frcnn_extra.net == "vgg16":
        fasterRCNN = vgg16(frcnn_extra.imdb_train.classes, pretrained=pretrained, class_agnostic=frcnn_extra.class_agnostic, model_path="", pth_path="data/pretrained_model/vgg16_caffe.pth")

    fasterRCNN.create_architecture()
    if pth_pretrained:
        checkpoint = torch.load(pth_pretrained)
        fasterRCNN.load_state_dict(checkpoint['model'])
    fasterRCNN = torch.load(source_pretrained)

    fasterRCNN = fasterRCNN.to(device)

    eval_fasterRCNN(cuda=device, model=fasterRCNN, frcnn_extra=frcnn_extra, is_break=False)



if __name__ == "__main__":
    main()
