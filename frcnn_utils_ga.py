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

class FasterRCNN_prepare_ga(FasterRCNN_prepare):
    def __init__(self, net, batch_size_train, s_dataset, t_dataset, lr_decay_step=5, lr_decay_gamma=0.1, cfg_file=None, debug=False):
        FasterRCNN_prepare.__init__(self, net, batch_size_train, s_dataset, lr_decay_step=lr_decay_step, lr_decay_gamma=lr_decay_gamma, cfg_file=cfg_file, debug=debug)
        self.t_dataset = t_dataset

    def ga_forward(self):

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

        random_ind = torch.randperm(t_dataset_train.data_size / 2)
        tracks_datasets = torch.utils.data.Subset(t_dataset_train, random_ind)
        self.tracks_dataloader_train = torch.utils.data.DataLoader(tracks_datasets, batch_size=self.batch_size_train,
                                                                sampler=t_sampler_batch, num_workers=0)


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
        self.tracks_iters_per_epoch = int(t_dataset_train.data_size / 2)

def train_frcnn_ga(alpha, frcnn_extra_ada, device, fasterRCNN, optimizer, is_debug=False):

    fasterRCNN.train()
    loss_temp = 0


    src_data_iter = iter(frcnn_extra_ada.s_dataloader_train)
    tar_data_iter = iter(frcnn_extra_ada.t_dataloader_train)
    tracks_data_iter = iter(frcnn_extra_ada.tracks_dataloader_train)

    if frcnn_extra_ada.t_iters_per_epoch < frcnn_extra_ada.s_iters_per_epoch:
        iters_per_epoch = frcnn_extra_ada.t_iters_per_epoch
    else:
        iters_per_epoch = frcnn_extra_ada.s_iters_per_epoch

    s_tt_base_feat = None
    t_tt_base_feat = None

    for step in range(iters_per_epoch):

        if step == frcnn_extra_ada.s_iters_per_epoch:
            src_data_iter = iter(frcnn_extra_ada.s_dataloader_train)
        if step == frcnn_extra_ada.t_iters_per_epoch:
            tar_data_iter = iter(frcnn_extra_ada.t_dataloader_train)
        if step == frcnn_extra_ada.tracks_iters_per_epoch:
            tar_data_iter = iter(frcnn_extra_ada.tracks_dataloader_train)

        tgt_data = next(tar_data_iter)
        track_data = next(tracks_data_iter)
        src_data = next(src_data_iter)
        src_im_data = src_data[0].to(device)
        src_im_info = src_data[1].to(device)
        src_gt_boxes = src_data[2].to(device)
        src_num_boxes = src_data[3].to(device)
        src_need_backprop = src_data[4].to(device)
        print("file:{}".format(src_data[5]))

        tgt_im_data = tgt_data[0].to(device)
        tgt_im_info = tgt_data[1].to(device)
        tgt_gt_boxes = tgt_data[2].to(device)
        tgt_num_boxes = tgt_data[3].to(device)
        tgt_need_backprop = tgt_data[4].to(device)

        track_im_data = tgt_data[0].to(device)
        track_im_info = tgt_data[1].to(device)
        track_gt_boxes = tgt_data[2].to(device)
        track_num_boxes = tgt_data[3].to(device)
        track_need_backprop = tgt_data[4].to(device)
        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN.forward(track_im_data, track_im_info, track_gt_boxes, track_num_boxes)
        track_loss = rpn_loss_cls + rpn_loss_bbox + RCNN_loss_cls + RCNN_loss_bbox
        track_loss.backward()

        g_m = {}
        for n, p in fasterRCNN.named_paremeters():
            g_m[n] = p.grad

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, DA_img_loss_cls, DA_ins_loss_cls, tgt_DA_img_loss_cls, tgt_DA_ins_loss_cls, \
        DA_cst_loss, tgt_DA_cst_loss, base_feat, tgt_base_feat, pooled_feat, tgt_pooled_feat = \
            fasterRCNN.forward_da(src_im_data, src_im_info, src_gt_boxes, src_num_boxes, src_need_backprop,
                       tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop, track_im_data, )

        s_loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

        loss = s_loss + alpha * (DA_img_loss_cls.mean() + DA_ins_loss_cls.mean() \
                               + tgt_DA_img_loss_cls.mean() + tgt_DA_ins_loss_cls.mean() + DA_cst_loss.mean() + tgt_DA_cst_loss.mean())
        loss_temp += loss.item()

        loss_DA_img_cls = alpha * (DA_img_loss_cls.item() + tgt_DA_img_loss_cls.item()) / 2
        loss_DA_ins_cls = alpha * (DA_ins_loss_cls.item() + tgt_DA_ins_loss_cls.item()) / 2
        loss_DA_cst = alpha * (DA_cst_loss.item() + tgt_DA_cst_loss.item()) / 2


        # backward
        optimizer.zero_grad()
        loss.backward()

        cs_loss = torch.zeros(len(g_m.keys()))


        for i, (n, p) in enumerate(fasterRCNN.named_paremeters()):
           cs_loss[i] = F.cosine_similarity(p.grad, g_m[n])

        cs_loss.norm(p=1).backward()

        if frcnn_extra_ada.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()

        if is_debug:
            break
    loss_temp = loss_temp
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label
    del DA_img_loss_cls
    del DA_ins_loss_cls
    del tgt_DA_img_loss_cls
    del tgt_DA_ins_loss_cls
    del DA_cst_loss
    del tgt_DA_cst_loss
    del loss
    return loss_temp, loss_DA_img_cls, loss_DA_ins_cls, loss_DA_cst

def train_frcnn_da(alpha, frcnn_extra_ada, device, fasterRCNN, optimizer, is_debug=False):



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

    for step in range(iters_per_epoch):

        if step == frcnn_extra_ada.s_iters_per_epoch:
            src_data_iter = iter(frcnn_extra_ada.s_dataloader_train)
        if step == frcnn_extra_ada.t_iters_per_epoch:
            tar_data_iter = iter(frcnn_extra_ada.t_dataloader_train)

        tgt_data = next(tar_data_iter)
        src_data = next(src_data_iter)
        src_im_data = src_data[0].to(device)
        src_im_info = src_data[1].to(device)
        src_gt_boxes = src_data[2].to(device)
        src_num_boxes = src_data[3].to(device)
        src_need_backprop = src_data[4].to(device)
        print("file:{}".format(src_data[5]))

        tgt_im_data = tgt_data[0].to(device)
        tgt_im_info = tgt_data[1].to(device)
        tgt_gt_boxes = tgt_data[2].to(device)
        tgt_num_boxes = tgt_data[3].to(device)
        tgt_need_backprop = tgt_data[4].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, DA_img_loss_cls, DA_ins_loss_cls, tgt_DA_img_loss_cls, tgt_DA_ins_loss_cls, \
        DA_cst_loss, tgt_DA_cst_loss, base_feat, tgt_base_feat, pooled_feat, tgt_pooled_feat = \
            fasterRCNN.forward_da(src_im_data, src_im_info, src_gt_boxes, src_num_boxes, src_need_backprop,
                       tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop)

        s_loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

        loss = s_loss + alpha * (DA_img_loss_cls.mean() + DA_ins_loss_cls.mean() \
                               + tgt_DA_img_loss_cls.mean() + tgt_DA_ins_loss_cls.mean() + DA_cst_loss.mean() + tgt_DA_cst_loss.mean())
        loss_temp += loss.item()

        loss_DA_img_cls = alpha * (DA_img_loss_cls.item() + tgt_DA_img_loss_cls.item()) / 2
        loss_DA_ins_cls = alpha * (DA_ins_loss_cls.item() + tgt_DA_ins_loss_cls.item()) / 2
        loss_DA_cst = alpha * (DA_cst_loss.item() + tgt_DA_cst_loss.item()) / 2

        # backward
        optimizer.zero_grad()
        loss.backward()
        if frcnn_extra_ada.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()

        if is_debug:
            break
    loss_temp = loss_temp
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label
    del DA_img_loss_cls
    del DA_ins_loss_cls
    del tgt_DA_img_loss_cls
    del tgt_DA_ins_loss_cls
    del DA_cst_loss
    del tgt_DA_cst_loss
    del loss
    return loss_temp, loss_DA_img_cls, loss_DA_ins_cls, loss_DA_cst

def train_frcnn_da_img(alpha, frcnn_extra_ada, device, fasterRCNN, optimizer, is_break=False):

    fasterRCNN.train()
    loss_temp = 0

    src_data_iter = iter(frcnn_extra_ada.s_dataloader_train)
    tar_data_iter = iter(frcnn_extra_ada.t_dataloader_train)

    if frcnn_extra_ada.t_iters_per_epoch < frcnn_extra_ada.s_iters_per_epoch:
        iters_per_epoch = frcnn_extra_ada.t_iters_per_epoch
    else:
        iters_per_epoch = frcnn_extra_ada.s_iters_per_epoch

    for step in range(iters_per_epoch):

        if step == frcnn_extra_ada.s_iters_per_epoch:
            src_data_iter = iter(frcnn_extra_ada.s_dataloader_train)
        if step == frcnn_extra_ada.t_iters_per_epoch:
            tar_data_iter = iter(frcnn_extra_ada.t_dataloader_train)

        tgt_data = next(tar_data_iter)
        src_data = next(src_data_iter)
        src_im_data = src_data[0].to(device)
        src_im_info = src_data[1].to(device)
        src_gt_boxes = src_data[2].to(device)
        src_num_boxes = src_data[3].to(device)
        src_need_backprop = src_data[4].to(device)
        print("file:{}".format(src_data[5]))

        tgt_im_data = tgt_data[0].to(device)
        tgt_im_info = tgt_data[1].to(device)
        tgt_gt_boxes = tgt_data[2].to(device)
        tgt_num_boxes = tgt_data[3].to(device)
        tgt_need_backprop = tgt_data[4].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, DA_img_loss_cls, DA_ins_loss_cls, tgt_DA_img_loss_cls, tgt_DA_ins_loss_cls, \
        DA_cst_loss, tgt_DA_cst_loss = \
            fasterRCNN.forward_da(src_im_data, src_im_info, src_gt_boxes, src_num_boxes, src_need_backprop,
                       tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop)

        s_loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

        loss = s_loss + alpha * (DA_img_loss_cls.mean() + 0 * DA_ins_loss_cls.mean() \
                               + tgt_DA_img_loss_cls.mean() + 0*tgt_DA_ins_loss_cls.mean() + 0*DA_cst_loss.mean() + 0*tgt_DA_cst_loss.mean())
        loss_temp += loss.item()

        loss_DA_img_cls = alpha * (DA_img_loss_cls.item() + tgt_DA_img_loss_cls.item()) / 2
        loss_DA_ins_cls = alpha * (DA_ins_loss_cls.item() + tgt_DA_ins_loss_cls.item()) / 2
        loss_DA_cst = alpha * (DA_cst_loss.item() + tgt_DA_cst_loss.item()) / 2

        # backward
        optimizer.zero_grad()
        loss.backward()
        if frcnn_extra_ada.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()

        if is_break:
            break
    loss_temp = loss_temp / frcnn_extra_ada.s_iters_per_epoch
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label
    del DA_img_loss_cls
    del DA_ins_loss_cls
    del tgt_DA_img_loss_cls
    del tgt_DA_ins_loss_cls
    del DA_cst_loss
    del tgt_DA_cst_loss
    del loss
    return loss_temp, loss_DA_img_cls, loss_DA_ins_cls, loss_DA_cst

def eval_frcnn_s_da(frcnn_extra, device, fasterRCNN, is_break=False):
    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(frcnn_extra.s_output_dir, 'detections.pkl')
    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    data_iter_test = iter(frcnn_extra.s_dataloader_test)
    for i in range(frcnn_extra.s_num_images_test):
        data_test = next(data_iter_test)
        im_data = data_test[0].to(device)
        im_info = data_test[1].to(device)
        gt_boxes = data_test[2].to(device)
        num_boxes = data_test[3].to(device)
        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

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
                    box_deltas = box_deltas.view(1, -1, 4 * len(frcnn_extra.s_imdb_test.classes))

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
        for j in range(1, frcnn_extra.s_imdb_test.num_classes):
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
                frcnn_extra.s_all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                frcnn_extra.s_all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if frcnn_extra.max_per_image > 0:
            image_scores = np.hstack([frcnn_extra.s_all_boxes[j][i][:, -1]
                                      for j in range(1, frcnn_extra.s_imdb_test.num_classes)])
            if len(image_scores) > frcnn_extra.max_per_image:
                image_thresh = np.sort(image_scores)[-frcnn_extra.max_per_image]
                for j in range(1, frcnn_extra.s_imdb_test.num_classes):
                    keep = np.where(frcnn_extra.s_all_boxes[j][i][:, -1] >= image_thresh)[0]
                    frcnn_extra.s_all_boxes[j][i] = frcnn_extra.s_all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        if is_break:
            break
    ap = frcnn_extra.s_imdb_test.evaluate_detections(frcnn_extra.s_all_boxes, frcnn_extra.s_output_dir)
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label

    return ap

def eval_frcnn_t_da(frcnn_extra, device, fasterRCNN, is_break=False):
    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(frcnn_extra.t_output_dir, 'detections.pkl')
    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    data_iter_test = iter(frcnn_extra.t_dataloader_test)
    for i in range(frcnn_extra.t_num_images_test):
        data_test = next(data_iter_test)
        im_data = data_test[0].to(device)
        im_info = data_test[1].to(device)
        gt_boxes = data_test[2].to(device)
        num_boxes = data_test[3].to(device)
        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

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
                    box_deltas = box_deltas.view(1, -1, 4 * len(frcnn_extra.t_imdb_test.classes))

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
        for j in range(1, frcnn_extra.t_imdb_test.num_classes):
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
                frcnn_extra.t_all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                frcnn_extra.t_all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if frcnn_extra.max_per_image > 0:
            image_scores = np.hstack([frcnn_extra.t_all_boxes[j][i][:, -1]
                                      for j in range(1, frcnn_extra.t_imdb_test.num_classes)])
            if len(image_scores) > frcnn_extra.max_per_image:
                image_thresh = np.sort(image_scores)[-frcnn_extra.max_per_image]
                for j in range(1, frcnn_extra.t_imdb_test.num_classes):
                    keep = np.where(frcnn_extra.t_all_boxes[j][i][:, -1] >= image_thresh)[0]
                    frcnn_extra.t_all_boxes[j][i] = frcnn_extra.t_all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        if is_break:
            break
    ap = frcnn_extra.t_imdb_test.evaluate_detections(frcnn_extra.t_all_boxes, frcnn_extra.t_output_dir)
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label

    return ap