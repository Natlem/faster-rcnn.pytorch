from roi_da_data_layer.roidb import combined_roidb
from roi_da_data_layer.roidb import combined_roidb
from roi_da_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
import numpy as np
from model.roi_layers import nms

from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import torch
import os
import time
from frcnn_utils import FasterRCNN_prepare, sampler

class FasterRCNN_prepare_another_da(FasterRCNN_prepare):
    def __init__(self, net, batch_size_train, s_dataset, t_dataset, lr_decay_step=5, lr_decay_gamma=0.1, cfg_file=None, debug=False):
        FasterRCNN_prepare.__init__(self, net, batch_size_train, s_dataset, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, cfg_file=cfg_file, debug=debug)
        self.t_dataset = t_dataset

    def tar_da_forward(self):

        #### Source loading
        self.forward()
        t_imdb_name, t_imdbval_name, set_cfgs = self.get_imdb_name(self.t_dataset)

        self.t_imdb_train, t_roidb_train, t_ratio_list_train, t_ratio_index_train = combined_roidb(t_imdb_name)
        self.t_train_size = len(t_roidb_train)
        self.t_imdb_test, t_roidb_test, t_ratio_list_test, t_ratio_index_test = combined_roidb(t_imdbval_name, False)
        self.t_imdb_test.competition_mode(on=True)

        output_dir = self.save_dir + "/" + self.net + "/" + self.t_dataset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        t_sampler_batch = sampler(self.t_train_size, self.batch_size_train)
        t_dataset_train = roibatchLoader(t_roidb_train, t_ratio_list_train, t_ratio_index_train, self.batch_size_train, \
                                           self.t_imdb_train.num_classes, training=True, is_target=True)
        self.t_dataloader_train = torch.utils.data.DataLoader(t_dataset_train, batch_size=self.batch_size_train,
                                                                sampler=t_sampler_batch, num_workers=0)

        save_name = 'faster_rcnn_{}'.format(self.net)
        self.t_num_images_test = len(self.t_imdb_test.image_index)
        self.t_all_boxes = [[[] for _ in range(self.t_num_images_test)]
                              for _ in range(self.t_imdb_test.num_classes)]
        self.t_output_dir = get_output_dir(self.t_imdb_test, save_name)
        t_dataset_test = roibatchLoader(t_roidb_test, t_ratio_list_test, t_ratio_index_test, self.batch_size_test, \
                                          self.t_imdb_test.num_classes, training=False, normalize=False, is_target=True)
        self.t_dataloader_test = torch.utils.data.DataLoader(t_dataset_test, batch_size=self.batch_size_test,
                                                               shuffle=False, num_workers=0,
                                                               pin_memory=True)

        self.t_iters_per_epoch = int(self.t_train_size / self.batch_size_train)


def train_da(alpha, epochs, totat_steps, frcnn_extra_ada, device, fasterRCNN, optimizer, img_domain_optimizer, inst_domain_optimizer, is_debug=False):
    fasterRCNN.train()
    loss_temp = 0


    src_data_iter = iter(frcnn_extra_ada.s_dataloader_train)
    tar_data_iter = iter(frcnn_extra_ada.t_dataloader_train)

    if frcnn_extra_ada.t_iters_per_epoch < frcnn_extra_ada.s_iters_per_epoch:
        iters_per_epoch = frcnn_extra_ada.t_iters_per_epoch
    else:
        iters_per_epoch = frcnn_extra_ada.s_iters_per_epoch

    s_tt_base_feat = None
    t_tt_base_feat = None
    total_train_size = epochs * frcnn_extra_ada.t_train_size

    for step in range(iters_per_epoch):

        if step == frcnn_extra_ada.s_iters_per_epoch:
            src_data_iter = iter(frcnn_extra_ada.s_dataloader_train)
        if step == frcnn_extra_ada.t_iters_per_epoch:
            tar_data_iter = iter(frcnn_extra_ada.t_dataloader_train)

        totat_steps += 1

        tgt_data = next(tar_data_iter)
        src_data = next(src_data_iter)

        src_im_data = src_data[0].to(device)
        src_im_info = src_data[1].to(device)
        src_gt_boxes = src_data[2].to(device)
        src_num_boxes = src_data[3].to(device)

        tgt_im_data = tgt_data[0].to(device)
        tgt_im_info = tgt_data[1].to(device)
        tgt_gt_boxes = tgt_data[2].to(device)
        tgt_num_boxes = tgt_data[3].to(device)

        src_rois, src_cls_prob, src_bbox_pred, \
        src_rpn_loss_cls, src_rpn_loss_box, \
        src_RCNN_loss_cls, src_RCNN_loss_bbox, \
        src_rois_label, src_base_feat, src_pooled_feat = fasterRCNN(src_im_data, im_info, src_gt_boxes, src_num_boxes)

        target_base_feat, target_pooled_feat = fasterRCNN(tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, target=True)

        img_domain_classifier.zero_grad()
        inst_domain_classifier.zero_grad()

        p = totat_steps / total_train_size
        beta = (2./(1.+np.exp(-10 * p))) - 1.


        src_img_feat = img_domain_classifier(src_base_feat, beta)
        target_img_feat = img_domain_classifier(target_base_feat, beta)
        src_inst_feat = inst_domain_classifier(src_pooled_feat, beta)
        target_inst_feat = inst_domain_classifier(target_pooled_feat, beta)

        src_img_loss = domain_loss(src_img_feat, 0) 
        tar_img_loss = domain_loss(target_img_feat, 1)
        src_inst_loss = domain_loss(src_inst_feat, 0)
        tar_inst_loss = domain_loss(target_inst_feat, 1)

        src_consistency_loss = consistency_loss(src_img_feat, src_inst_feat)
        tar_consistency_loss = consistency_loss(target_img_feat, target_inst_feat)


        loss = src_rpn_loss_cls.mean() + src_rpn_loss_box.mean() \
           + src_RCNN_loss_cls.mean() + src_RCNN_loss_bbox.mean() \
           + 0.1*(src_img_loss.mean() + src_inst_loss.mean()) \
           + 0.1*(tar_img_loss.mean() + tar_inst_loss.mean()) \
           + 0.1*(src_consistency_loss.mean() + tar_consistency_loss.mean())
        loss_temp += loss.item()
        supervised_loss += (src_rpn_loss_cls.mean() + src_rpn_loss_box.mean() \
           + src_RCNN_loss_cls.mean() + src_RCNN_loss_bbox.mean()).item()
        domain_loss_img = (src_img_loss.mean() + tar_img_loss.mean()).item()
        domain_loss_inst = (src_inst_loss.mean() + tar_inst_loss.mean()).item()
        domain_loss_cons = (src_consistency_loss.mean() + tar_consistency_loss.mean()).item()


        optimizer.zero_grad()
        img_domain_optimizer.zero_grad()
        inst_domain_optimizer.zero_grad()
        loss.backward()
        if args.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()

    del src_rois
    del src_cls_prob
    del src_bbox_pred
    del src_rpn_loss_cls
    del src_rpn_loss_box
    del src_RCNN_loss_cls
    del src_RCNN_loss_bbox
    del src_rois_label
    del src_img_loss
    del tar_img_loss
    del src_inst_loss
    del tar_inst_loss
    del src_consistency_loss
    del tar_consistency_loss
    del loss

    return 