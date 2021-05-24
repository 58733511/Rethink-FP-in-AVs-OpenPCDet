#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:57:59 2021

@author: leichen
"""
import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file

import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


    

cfg_from_yaml_file("/home/leichen/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml", cfg)
log_file="/home/leichen/OpenPCDet/tools/output.txt"
logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)


test_set, dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=1, logger=logger, training=False
    )
dataset = dataloader.dataset
class_names = dataset.class_names
metric = {
    'gt_num': 0,
}
for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    metric['recall_roi_%s' % str(cur_thresh)] = 0
    metric['recall_rcnn_%s' % str(cur_thresh)] = 0

dataset = dataloader.dataset
class_names = dataset.class_names
det_annos = []


ret_dict = {}

gt_num_cnt = metric['gt_num']
for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
    logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
    ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
    ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

total_pred_objects = 0
for anno in det_annos:
    total_pred_objects += anno['name'].__len__()
logger.info('Average predicted number of objects(%d samples): %.3f'
            % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

with open('/home/leichen/CCU_new_cluster/result/result_09_05/editted_result.pkl', 'rb') as f:
    det_annos = pickle.load(f)

result_str, result_dict = dataset.evaluation(
    det_annos, class_names,
    eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    output_path="/home/leichen/OpenPCDet/output/"
)
logger.info(result_str)

ret_dict.update(result_dict)




