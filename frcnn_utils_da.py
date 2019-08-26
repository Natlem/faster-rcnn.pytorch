import sys

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
import torch.nn.functional as F
import torch
import os
import time
from model.faster_rcnn.domain_adapt import D_cls_image, D_cls_inst, consistency_reg
import torchvision
from frcnn.frcnn_utils import FasterRCNN_prepare, sampler

class FasterRCNN_prepare_da(FasterRCNN_prepare):
    def __init__(self, net, batch_size_train, src_dataset, tar_dataset, cfg_file=None):
        FasterRCNN_prepare.__init__(self, net, batch_size_train, src_dataset, cfg_file)
        self.tar_dataset = tar_dataset

    def tar_da_forward(self):
        self.forward()
        tar_imdb_name, tar_imdbval_name, set_cfgs = self.get_imdb_name(self.tar_dataset)
        
        self.tar_imdb_train, roidb_train, ratio_list_train, ratio_index_train = combined_roidb(tar_imdb_name)
        self.tar_train_size = len(roidb_train)
        self.tar_imdb_test, roidb_test, ratio_list_test, ratio_index_test = combined_roidb(tar_imdbval_name, False)
        self.tar_imdb_test.competition_mode(on=True)


        output_dir = self.save_dir + "/" + self.net + "/" + self.tar_dataset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tar_sampler_batch = sampler(self.tar_train_size, self.batch_size_train)
        tar_dataset_train = roibatchLoader(roidb_train, ratio_list_train, ratio_index_train, self.batch_size_train, \
                                       self.tar_imdb_train.num_classes, training=True)
        self.tar_dataloader_train = torch.utils.data.DataLoader(tar_dataset_train, batch_size=self.batch_size_train,
                                                       sampler=tar_sampler_batch, num_workers=0)

        save_name = 'faster_rcnn_{}'.format(self.net)
        self.tar_num_images_test = len(self.tar_imdb_test.image_index)
        self.tar_all_boxes = [[[] for _ in range(self.tar_num_images_test)]
                     for _ in range(self.tar_imdb_test.num_classes)]
        self.tar_output_dir = get_output_dir(self.tar_imdb_test, save_name)
        tar_dataset_test = roibatchLoader(roidb_test, ratio_list_test, ratio_index_test, self.batch_size_test, \
                                      self.tar_imdb_test.num_classes, training=False, normalize=False)
        self.tar_dataloader_test = torch.utils.data.DataLoader(tar_dataset_test, batch_size=self.batch_size_test,
                                                      shuffle=False, num_workers=0,
                                                      pin_memory=True)

        self.tar_iters_per_epoch = int(self.tar_train_size / self.batch_size_train)

def eval_frcnn_da(frcnn_extra, device, fasterRCNN, is_break=False):
    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(frcnn_extra.tar_output_dir, 'detections.pkl')
    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    data_iter_test = iter(frcnn_extra.tar_dataloader_test)
    for i in range(frcnn_extra.tar_num_images_test):
        data_test = next(data_iter_test)
        im_data = data_test[0].to(device)
        im_info = data_test[1].to(device)
        gt_boxes = data_test[2].to(device)
        num_boxes = data_test[3].to(device)
        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, _, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if frcnn_extra.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(frcnn_extra.tar_imdb_test.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data_test[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        for j in range(1, frcnn_extra.tar_imdb_test.num_classes):
            inds = torch.nonzero(scores[:, j] > frcnn_extra.thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if frcnn_extra.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                frcnn_extra.tar_all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                frcnn_extra.tar_all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if frcnn_extra.max_per_image > 0:
            image_scores = np.hstack([frcnn_extra.tar_all_boxes[j][i][:, -1]
                                      for j in range(1, frcnn_extra.tar_imdb_test.num_classes)])
            if len(image_scores) > frcnn_extra.max_per_image:
                image_thresh = np.sort(image_scores)[-frcnn_extra.max_per_image]
                for j in range(1, frcnn_extra.tar_imdb_test.num_classes):
                    keep = np.where(frcnn_extra.tar_all_boxes[j][i][:, -1] >= image_thresh)[0]
                    frcnn_extra.tar_all_boxes[j][i] = frcnn_extra.tar_all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        if is_break:
            break
    ap = frcnn_extra.tar_imdb_test.evaluate_detections(frcnn_extra.tar_all_boxes, frcnn_extra.tar_output_dir)
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label

    return ap

def d_criteria(d_x, label):
    loss = F.cross_entropy(d_x, label)
    return loss

def train_frcnn_da(frcnn_extra_da, device, fasterRCNN, optimizer, d_cls_image, d_cls_inst, d_image_opt, d_inst_opt,
                   start_steps, total_steps, is_break=False):



    fasterRCNN.train()
    d_cls_image.train()
    d_cls_inst.train()
    loss_temp = 0


    src_data_iter = iter(frcnn_extra_da.dataloader_train)
    tar_data_iter = iter(frcnn_extra_da.tar_dataloader_train)

    if frcnn_extra_da.tar_iters_per_epoch < frcnn_extra_da.iters_per_epoch:
        iter_per_epoch = frcnn_extra_da.tar_iters_per_epoch
    else:
        iter_per_epoch = frcnn_extra_da.iters_per_epoch

    for step in range(iter_per_epoch):
        tar_data = next(tar_data_iter)
        src_data = next(src_data_iter)
        src_im_data = src_data[0].to(device)
        src_im_info = src_data[1].to(device)
        src_gt_boxes = src_data[2].to(device)
        src_num_boxes = src_data[3].to(device)



        tar_im_data = tar_data[0].to(device)

        if tar_im_data.shape[2:] != src_im_data.shape[2:]:
            tar_im_data = F.interpolate(tar_im_data, size=src_im_data.shape[2:]).to(device)

        tar_im_info = tar_data[1].to(device)
        tar_gt_boxes = None
        tar_num_boxes = None

        fasterRCNN.zero_grad()
        d_cls_image.zero_grad()
        d_cls_inst.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, src_feat_map, src_roi_pool = fasterRCNN(src_im_data, src_im_info, src_gt_boxes, src_num_boxes, is_target=False)
        tar_feat_map, tar_roi_pool = fasterRCNN(tar_im_data, tar_im_info, tar_gt_boxes, tar_num_boxes, is_target=True)

        p = float(step + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        d_cls_image.set_beta(constant)
        d_cls_inst.set_beta(constant)

        src_d_img_score = d_cls_image(src_feat_map)
        src_d_inst_score = d_cls_inst(src_roi_pool)
        tar_d_img_score = d_cls_image(tar_feat_map)
        tar_d_inst_score = d_cls_inst(tar_roi_pool)

        s1 = list(src_d_img_score.size())[0]
        s2 = list(tar_d_img_score.size())[0]
        s3 = list(src_d_inst_score.size())[0]
        s4 = list(tar_d_inst_score.size())[0]

        src_img_label = torch.zeros(s1).long().to(device)
        src_inst_label = torch.zeros(s3).long().to(device)
        tar_img_label = torch.ones(s2).long().to(device)
        tar_inst_label = torch.ones(s4).long().to(device)

        src_d_img_loss = d_criteria(src_d_img_score, src_img_label)
        src_d_inst_loss = d_criteria(src_d_inst_score, src_inst_label)
        tar_d_img_loss = d_criteria(tar_d_img_score, tar_img_label)
        tar_d_inst_loss = d_criteria(tar_d_inst_score, tar_inst_label)

        d_img_loss = src_d_img_loss + tar_d_img_loss
        d_inst_loss = src_d_inst_loss + tar_d_inst_loss

        src_feat_map_dim = list(src_feat_map.size())[1] * list(src_feat_map.size())[2] * list(src_feat_map.size())[3]
        tar_feat_map_dim = list(tar_feat_map.size())[1] * list(tar_feat_map.size())[2] * list(tar_feat_map.size())[3]

        src_d_cst_loss = consistency_reg(src_feat_map_dim, src_d_img_score, src_d_inst_score, domain='src')
        tar_d_cst_loss = consistency_reg(tar_feat_map_dim, tar_d_img_score, tar_d_inst_score, domain='tar')

        d_cst_loss = src_d_cst_loss + tar_d_cst_loss

        s_loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss= 0 * s_loss + (0.1)*(d_img_loss.mean() + d_inst_loss.mean() + d_cst_loss.mean())
        loss_temp += float(loss.item())
        optimizer.zero_grad()
        d_inst_opt.zero_grad()
        d_image_opt.zero_grad()
        loss.backward()
        if frcnn_extra_da.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()
        d_inst_opt.step()
        d_image_opt.step()
        if is_break:
            break
    loss_temp = loss_temp / frcnn_extra_da.iters_per_epoch
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label
    del src_d_img_loss
    del src_d_inst_loss
    del tar_d_img_loss
    del tar_d_inst_loss
    del loss
    return loss_temp, d_cst_loss, d_img_loss, d_inst_loss

def train_frcnn_da_img(frcnn_extra_da, device, fasterRCNN, optimizer, d_cls_image, d_image_opt,
                   start_steps, total_steps, is_break=False):



    fasterRCNN.train()
    d_cls_image.train()
    loss_temp = 0


    src_data_iter = iter(frcnn_extra_da.dataloader_train)
    tar_data_iter = iter(frcnn_extra_da.tar_dataloader_train)

    if frcnn_extra_da.tar_iters_per_epoch < frcnn_extra_da.iters_per_epoch:
        iter_per_epoch = frcnn_extra_da.tar_iters_per_epoch
    else:
        iter_per_epoch = frcnn_extra_da.iters_per_epoch

    for step in range(iter_per_epoch):
        tar_data = next(tar_data_iter)
        src_data = next(src_data_iter)
        src_im_data = src_data[0].to(device)
        src_im_info = src_data[1].to(device)
        src_gt_boxes = src_data[2].to(device)
        src_num_boxes = src_data[3].to(device)



        tar_im_data = tar_data[0].to(device)

        if tar_im_data.shape[2:] != src_im_data.shape[2:]:
            tar_im_data = F.interpolate(tar_im_data, size=src_im_data.shape[2:]).to(device)

        tar_im_info = tar_data[1].to(device)
        tar_gt_boxes = None
        tar_num_boxes = None

        fasterRCNN.zero_grad()
        d_cls_image.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, src_feat_map, src_roi_pool = fasterRCNN(src_im_data, src_im_info, src_gt_boxes, src_num_boxes, is_target=False)
        tar_feat_map, tar_roi_pool = fasterRCNN(tar_im_data, tar_im_info, tar_gt_boxes, tar_num_boxes, is_target=True)

        p = float(step + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        d_cls_image.set_beta(constant)

        src_d_img_score = d_cls_image(src_feat_map)
        tar_d_img_score = d_cls_image(tar_feat_map)

        s1 = list(src_d_img_score.size())[0]
        s2 = list(tar_d_img_score.size())[0]

        src_img_label = torch.zeros(s1).long().to(device)
        tar_img_label = torch.ones(s2).long().to(device)

        src_d_img_loss = d_criteria(src_d_img_score, src_img_label)
        tar_d_img_loss = d_criteria(tar_d_img_score, tar_img_label)

        d_img_loss = src_d_img_loss + tar_d_img_loss
        src_feat_map_dim = list(src_feat_map.size())[1] * list(src_feat_map.size())[2] * list(src_feat_map.size())[3]
        tar_feat_map_dim = list(tar_feat_map.size())[1] * list(tar_feat_map.size())[2] * list(tar_feat_map.size())[3]

        s_loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss=  s_loss + (0.1)*(d_img_loss.mean())
        loss_temp += float(loss.item())
        optimizer.zero_grad()
        d_image_opt.zero_grad()
        loss.backward()
        if frcnn_extra_da.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()
        d_image_opt.step()
        if is_break:
            break
    loss_temp = loss_temp / frcnn_extra_da.iters_per_epoch
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label
    del src_d_img_loss
    del tar_d_img_loss
    del loss
    return loss_temp, 0, d_img_loss, 0
