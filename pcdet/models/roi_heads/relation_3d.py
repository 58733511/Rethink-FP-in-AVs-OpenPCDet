#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:13:16 2021

@author: leichen
"""
import torch
def boardcase_sub (x):
    delta_x = torch.t(x) - torch.t(x).view(1, -1)
    return delta_x

def extract_position_matrix(rois, dim_g=64, wave_len=1000):
    #####rois: (B, num_rois, 7 + C)
    """ Extract position matrix
    Args:
        bbox: [num_boxes, 4]
    Returns:
        position_matrix: [num_boxes, nongt_dim, 4]
    """
    # change
    #xmin, ymin, xmax, ymax, zmin, zmax = mx.sym.split(data=bbox,
    #                                      num_outputs=6, axis=1)
    # [num_fg_classes, num_boxes, 1]
    bbox = rois.view(-1, rois.shape[-1])
    h, w, l, x, y, z, y_pi = torch.chunk(torch.t(bbox), 7, dim=0) 
    # [num_fg_classes, num_boxes, num_boxes]

    #  x in the range [0,70.4], y in the range [-40,40], z in the range [-3,1]
    delta_x = boardcase_sub (x)/ 70.4
    delta_y = boardcase_sub (y)/ 80
    delta_z = boardcase_sub (z)/ 4
    delta_h = torch.log(torch.t(h) / torch.t(h).view(1, -1))
    delta_w = torch.log(torch.t(w) / torch.t(w).view(1, -1))
    delta_l = torch.log(torch.t(l) / torch.t(l).view(1, -1))
    delta_pi = boardcase_sub (y_pi)/ 3.1415927410125732

    size = delta_h.size()
    print("size",size)
    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_z = delta_z.view(size[0], size[1], 1)
    delta_w = delta_w.view(size[0], size[1], 1)
    delta_h = delta_h.view(size[0], size[1], 1)
    delta_l = delta_l.view(size[0], size[1], 1)
    delta_pi = delta_pi.view(size[0], size[1], 1)


    position_mat = torch.cat((delta_x, delta_y, delta_z, delta_w, delta_h, delta_l, delta_pi), -1)
    position_mat = position_mat.view(size[0], size[1], 7)
    

    
    feat_range = torch.arange(dim_g / 8)
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1).cuda()
    position_mat = position_mat.view(size[0], size[1], 7, -1)
    position_mat = 100. * position_mat
    position_mat = position_mat.cuda()
    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    return embedding
