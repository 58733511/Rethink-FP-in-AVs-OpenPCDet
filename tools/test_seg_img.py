#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:43:37 2021

@author: leichen
"""
import cv2
import numpy as np
data_idx = "001000"
Seg_Img = cv2.imread('/media/leichen/SeagateBackupPlusDrive/KITTI_DATABASE_SEG/color_mask_3/pred_3_channel_'+str(data_idx)+'.png')

pts_seg_info = Seg_Img[-100,-100]
np_0 = np.array([[2,3,1,0],[2,3,1,-1]])
mask = np.where(np_0<0)
print(mask[0].shape)