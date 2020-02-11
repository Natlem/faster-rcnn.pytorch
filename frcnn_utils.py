import sys

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, \
    save_checkpoint, clip_gradient, vis_detections
from model.utils.blob import im_list_to_blob
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
import mmd
import cv2

class LoggerForSacred():
    def __init__(self, visdom_logger, ex_logger=None):
        self.visdom_logger = visdom_logger
        self.ex_logger = ex_logger


    def log_scalar(self, metrics_name, value, step):
        self.visdom_logger.scalar(metrics_name, step, [value])
        if self.ex_logger is not None:
            self.ex_logger.log_scalar(metrics_name, value, step)


def save_state_dict(fasterRCNN, optimizer, class_agnostic, save_name):
    save_checkpoint({
        'model': fasterRCNN.state_dict(),
        'optimizer': optimizer.state_dict(),
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': class_agnostic,
    }, save_name)

def save_conf(frcnn_extra, save_name):
    f = open(save_name, 'w')
    f.write("{}:{}\n".format('lr_decay_step', frcnn_extra.lr_decay_step))
    f.write("{}:{}\n".format('lr_decay_gamma', frcnn_extra.lr_decay_gamma))
    f.write("{}:{}\n".format('max_per_image', frcnn_extra.max_per_image))
    f.write("{}:{}\n".format('class_agnostic', frcnn_extra.class_agnostic))
    f.write("{}:{}\n".format('dataset', frcnn_extra.s_dataset))
    f.write("{}:{}\n".format('net', frcnn_extra.net))
    f.write("{}:{}\n".format('batch_size_train', frcnn_extra.batch_size_train))
    f.close()


class FasterRCNN_prepare():
    def __init__(self, net, batch_size_train, dataset, cfg_file=None, debug=False):
        self.lr_decay_step = 5
        self.lr_decay_gamma = 0.1
        self.max_per_image = 100
        self.thresh = 0.0 #0.0 for computing score, change to higher for visualization
        self.class_agnostic = False
        self.save_dir = "myTmp"
        self.large_scale = False
        self.cfg_file = cfg_file
        self.s_dataset = dataset
        self.net = net
        self.batch_size_train = batch_size_train
        self.batch_size_test = 1
        self.debug = debug



    def forward(self, is_training=True, is_da=False):
        s_imdb_name, s_imdbval_name, set_cfgs = self.get_imdb_name(self.s_dataset)

        if self.cfg_file is None:
            self.cfg_file = "frcnn/cfgs/{}_ls.yml".format(self.net) if self.large_scale else "frcnn/cfgs/{}.yml".format(self.net)
        if self.cfg_file is not None:
            cfg_from_file(self.cfg_file)
        if set_cfgs is not None:
            cfg_from_list(set_cfgs)

        cfg.TRAIN.USE_FLIPPED = True
        cfg.CUDA = True
        cfg.USE_GPU_NMS = True

        output_dir = self.save_dir + "/" + self.net + "/" + self.s_dataset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_name = 'faster_rcnn_{}'.format(self.net)

        if is_training:
            self.s_imdb_train, s_roidb_train, s_ratio_list_train, s_ratio_index_train = combined_roidb(s_imdb_name)
            self.s_train_size = len(s_roidb_train)


            s_sampler_batch = sampler(self.s_train_size, self.batch_size_train)

            s_dataset_train = roibatchLoader(s_roidb_train, s_ratio_list_train, s_ratio_index_train,
                                             self.batch_size_train, \
                                             self.s_imdb_train.num_classes, training=True)

            self.s_dataloader_train = torch.utils.data.DataLoader(s_dataset_train, batch_size=self.batch_size_train,
                                                                  sampler=s_sampler_batch, num_workers=0)
            self.s_iters_per_epoch = int(self.s_train_size / self.batch_size_train)

            self.s_imdb_test, s_roidb_test, s_ratio_list_test, s_ratio_index_test = combined_roidb(s_imdbval_name,
                                                                                                   False)
        self.s_imdb_test.competition_mode(on=True)
        self.s_num_images_test = len(self.s_imdb_test.image_index)
        self.s_all_boxes = [[[] for _ in range(self.s_num_images_test)]
                     for _ in range(self.s_imdb_test.num_classes)]
        self.s_output_dir = get_output_dir(self.s_imdb_test, save_name)


        dataset_test = roibatchLoader(s_roidb_test, s_ratio_list_test, s_ratio_index_test, self.batch_size_test, \
                                      self.s_imdb_test.num_classes, training=False, normalize=False)
        self.s_dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size_test,
                                                      shuffle=False, num_workers=0,
                                                      pin_memory=True)



    def get_imdb_name(self, dataset):
        if dataset == "pascal_voc":
            imdb_name = "voc_2007_trainval"
            imdbval_name = "voc_2007_test"
            set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        if dataset == "pascal_voc_person":
            imdb_name = "vocp_2007_trainval"
            imdbval_name = "vocp_2007_test"
            set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif dataset == "pascal_voc_0712":
            imdb_name = "voc_2007_trainval+voc_2012_trainval"
            imdbval_name = "voc_2007_test"
            set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif dataset == "hollywood":
            imdb_name = "hollywood_trainval"
            if self.debug:
                imdb_name = "hollywood_debug"
            imdbval_name = "hollywood_test"
            set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif dataset == "scuta":
            imdb_name = "scuta_trainval"
            if self.debug:
                imdb_name = "scuta_debug"
            imdbval_name = "scuta_test"
            set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif dataset == "scutb":
            imdb_name = "scutb_trainval"
            if self.debug:
                imdb_name = "scutb_debug"
            imdbval_name = "scutb_test"
            set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif dataset == "scuta_ori":
            #imdb_name = "scuta_ori_debug"
            imdb_name = "scuta_ori_trainval"
            imdbval_name = "scuta_ori_test"
            set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        return imdb_name, imdbval_name, set_cfgs

class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

def eval_frcnn(frcnn_extra, device, fasterRCNN, is_break=False):
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

def train_frcnn(frcnn_extra, device, fasterRCNN, optimizer, is_break=False):

    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    data_iter = iter(frcnn_extra.s_dataloader_train)
    for step in range(frcnn_extra.s_iters_per_epoch):
        data = next(data_iter)
        im_data = data[0].to(device)
        im_info = data[1].to(device)
        gt_boxes = data[2].to(device)
        num_boxes = data[3].to(device)

        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, _, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, is_target=False)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        #if (torch.isnan(loss)):
        #    print("NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        loss_temp += float(loss.item())
        optimizer.zero_grad()
        loss.backward()
        if frcnn_extra.net == "vgg16":
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()
        if is_break:
            break
    loss_temp = loss_temp / frcnn_extra.s_iters_per_epoch
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label
    del loss
    return loss_temp

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def eval_one_img_frcnn(one_image, frcnn_extra, device, fasterRCNN):

    #rgb to bgr
    im = one_image[:, :, ::-1]

    blobs, im_scales = _get_image_blob(im)
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data = im_data_pt.to(device)
    im_info = im_info_pt.to(device)
    gt_boxes = torch.zeros(1,1,5).to(device)
    num_boxes = torch.zeros(1).to(device)

    fasterRCNN.eval()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, feat_map_1, roi_pool_1 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, is_target=False)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).to(device) \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
            box_deltas = box_deltas.view(1, -1, 4 * frcnn_extra.s_imdb_train.num_classes)

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    im2show = np.copy(im)
    for j in range(1, frcnn_extra.s_imdb_train.num_classes):
        inds = torch.nonzero(scores[:, j] > frcnn_extra.thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            im2show = vis_detections(im2show, "", cls_dets.cpu().numpy(), 0.5)

    return im2show



def eval_kl_frcnn(frcnn_extra, device, fasterRCNN_1, fasterRCNN_2, is_break=False):

    fasterRCNN_1.train()
    fasterRCNN_2.train()
    loss_temp = 0
    kl_div_feat_temp = 0.
    kl_div_roi_temp = 0.

    mmd_feat_tmp = 0.
    mmd_inst_tmp = 0.

    data_iter = iter(frcnn_extra.s_dataloader_train)
    for step in range(frcnn_extra.s_iters_per_epoch):
        data = next(data_iter)
        im_data = data[0].to(device)
        im_info = data[1].to(device)
        gt_boxes = data[2].to(device)
        num_boxes = data[3].to(device)

        fasterRCNN_1.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, feat_map_1, roi_pool_1 = fasterRCNN_1(im_data, im_info, gt_boxes, num_boxes, is_target=False)

        fasterRCNN_2.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, feat_map_2, roi_pool_2 = fasterRCNN_2(im_data, im_info, gt_boxes, num_boxes, is_target=False)

        kl_div_feat_temp += torch.abs(F.kl_div(feat_map_1, feat_map_2)).item()
        kl_div_roi_temp += torch.abs(F.kl_div(roi_pool_1, roi_pool_2)).item()

        mmd_feat_tmp += mmd.mmd_loss(feat_map_1.view(1, -1), feat_map_2.view(1, -1))
        mmd_inst_tmp += mmd.mmd_loss(roi_pool_1, roi_pool_2)

        if is_break:
            break
    loss_temp = loss_temp / frcnn_extra.s_iters_per_epoch
    del rois
    del cls_prob
    del bbox_pred
    del rpn_loss_cls
    del rpn_loss_box
    del RCNN_loss_cls
    del RCNN_loss_bbox
    del rois_label

    return kl_div_feat_temp / frcnn_extra.s_iters_per_epoch, kl_div_roi_temp / frcnn_extra.s_iters_per_epoch, \
           mmd_feat_tmp / frcnn_extra.s_iters_per_epoch, mmd_inst_tmp / frcnn_extra.s_iters_per_epoch

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

    src_data_iter = iter(frcnn_extra_da.s_dataloader_train)
    tar_data_iter = iter(frcnn_extra_da.tar_dataloader_train)

    if frcnn_extra_da.tar_iters_per_epoch < frcnn_extra_da.s_iters_per_epoch:
        iter_per_epoch = frcnn_extra_da.tar_iters_per_epoch
    else:
        iter_per_epoch = frcnn_extra_da.s_iters_per_epoch

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
        rois_label, src_feat_map, src_roi_pool = fasterRCNN(src_im_data, src_im_info, src_gt_boxes, src_num_boxes,
                                                            is_target=False)
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
        loss = 1 * s_loss + (0.1) * (d_img_loss.mean() + d_inst_loss.mean() + d_cst_loss.mean())
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
    loss_temp = loss_temp / frcnn_extra_da.s_iters_per_epoch
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
        rois_label, src_feat_map, src_roi_pool = fasterRCNN(src_im_data, src_im_info, src_gt_boxes, src_num_boxes,
                                                            is_target=False)
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
        loss = s_loss + (0.1) * (d_img_loss.mean())
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
