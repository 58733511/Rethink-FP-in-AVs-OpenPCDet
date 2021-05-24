#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 11:00:51 2021

@author: leichen
"""
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np
import os
from typing import Tuple
from scipy import optimize
import numpy as np
from OpenGL.GL import glLineWidth
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numba
import matplotlib.pyplot as plt
import random
import argparse
import cv2
# from det3d.visualization.simplevis import kitti_vis,kitti_vis_pred,kitti_vis_pred_with_baseline
import pandas as pd
import torch
# from det3d.core import box_np_ops
import imageio
import pickle


class plot3d(object):
    def __init__(self):
        self.app = pg.mkQApp()
        self.view = gl.GLViewWidget()
        coord = gl.GLAxisItem()
        glLineWidth(3)
        coord.setSize(3, 3, 3)
        self.view.addItem(coord)

    def add_points(self, points, colors,sizes=2):
        points_item = gl.GLScatterPlotItem(pos=points, size=sizes, color=colors)
        self.view.addItem(points_item)

    def add_line_blue(self, p1, p2):
        lines = np.array([[p1[0], p1[1], p1[2]], [p2[0], p2[1], p2[2]]])
        lines_item = gl.GLLinePlotItem(
            pos=lines, mode="lines", color=(0, 0, 1, 1), width=3, antialias=True
        )
        self.view.addItem(lines_item)
    def add_line(self, p1, p2):
        lines = np.array([[p1[0], p1[1], p1[2]], [p2[0], p2[1], p2[2]]])
        lines_item = gl.GLLinePlotItem(
            pos=lines, mode="lines", color=(1, 0, 0, 1), width=3, antialias=True
        )
        self.view.addItem(lines_item)
    def show(self):
        self.view.show()
        self.app.exec()

def spherical_project(data_idx,rois,global_roi_grid_points):
    global batch_dict
    proj_H = 64
    proj_W = 2048
    proj_fov_up = 3
    proj_fov_down = -25.0
    points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission
      
    proj_pgm = np.full((proj_H, proj_W, 5), -1,
                              dtype=np.float32)
    proj_pgm1 = np.full((proj_H, proj_W, 5), -1,
                              dtype=np.float32)
    
    # for each point, where it is in the range image
    proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
    proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y
    data_idx=str(data_idx).replace("[","")
    data_idx=data_idx.replace("]","")
    data_idx=data_idx.replace("'","")

    scan = np.fromfile("/media/leichen/SeagateBackupPlusDrive/KITTI_DATABASE/training/velodyne/"+(data_idx)+".bin",dtype= np.float32)
    scan = scan.reshape((-1, 4))
    
    
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    
    fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
    
    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)
    
    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    
    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    # pdb.set_trace()
    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]
    
    # scale to image size using angular resolution
    proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]
    
    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    proj_x_copy = np.copy(proj_x)  # store a copy in orig order
    
    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    proj_y_copy = np.copy(proj_y)  # stope a copy in original order
    
    proj_pgm[proj_y,proj_x,0:3]  = points
    proj_pgm[proj_y,proj_x,3]  = remissions
    proj_pgm[proj_y,proj_x,4]  = depth
    for i in range (global_roi_grid_points.shape[0]):
        print("roi_scores",float(batch_dict["roi_scores"][0][i]))
        print("roi",batch_dict["rois"][0][i][:7])

        grid_points=global_roi_grid_points[i,:,:]
        print("grid_points",grid_points)
        depth_grid_points = np.linalg.norm(grid_points, 2, axis=1)
        min_depth=np.min(depth_grid_points)
        max_depth=np.max(depth_grid_points)

        grid_points_2d=np.zeros_like(grid_points)
        grid_points_2d[:,0]=-np.arctan2(grid_points[:,1], grid_points[:,0])
        grid_points_2d[:,1]=np.arcsin(grid_points[:,2] / depth_grid_points)

        # pdb.set_trace()
        # get projections in image coords
        grid_points_2d[:,0] = 0.5 * (grid_points_2d[:,0] / np.pi + 1.0)          # in [0.0, 1.0]
        grid_points_2d[:,1] = 1.0 - (grid_points_2d[:,1] + abs(fov_down)) / fov        # in [0.0, 1.0]
        
        # scale to image size using angular resolution
        grid_points_2d[:,0] *= proj_W                              # in [0.0, W]
        grid_points_2d[:,1] *= proj_H                              # in [0.0, H]
        grid_points_2d[:,0] = grid_points_2d[:,0]-768
        grid_points_2d_color=np.zeros_like(grid_points_2d[:,0])
        plt.imshow(proj_pgm[:,768:1281,4])
        for j in range(grid_points_2d.shape[0]):
            if (proj_pgm[int(round(grid_points_2d[j,1])),int(round(grid_points_2d[j,0]+768)),4]>min_depth) and (proj_pgm[int(round(grid_points_2d[j,1])),int(round(grid_points_2d[j,0]))+768,4]< max_depth):
                grid_points_2d_color[j]=0.6
            elif proj_pgm[int(round(grid_points_2d[j,1])),int(round(grid_points_2d[j,0]+768)),4]==-1:
                grid_points_2d_color[j]=0
            elif proj_pgm[int(round(grid_points_2d[j,1])),int(round(grid_points_2d[j,0]+768)),4]<min_depth:
                grid_points_2d_color[j]=0.3
            elif proj_pgm[int(round(grid_points_2d[j,1])),int(round(grid_points_2d[j,0]+768)),4]>max_depth:
                grid_points_2d_color[j]=1
        plt.scatter(grid_points_2d[:,0], grid_points_2d[:,1], s=1,c=grid_points_2d_color,cmap="Reds")
        plt.show()
        p3d = plot3d()
        pc = scan[:, 0:3] 
        pc_color = np.ones_like(scan[:, 0:3] )
        p3d.add_points(pc, pc_color)
        print("i",i)
        grid_points=global_roi_grid_points[i,:,:]
        grid_points_3d_color=np.zeros((grid_points_2d_color.shape[0],3))
        grid_points_3d_color[:,0]=0.5
        for j in range(1,3):
            grid_points_3d_color[:,j]=grid_points_2d_color
        p3d.add_points(grid_points, grid_points_3d_color,sizes=8)

        p3d.show()
    # p3d = plot3d()
    # pc = scan[:, 0:3] 
    # pc_color = np.ones_like(scan[:, 0:3] )
    # #print("pc_color",pc_color)
    # p3d.add_points(pc, pc_color)
    
    # grid_points=global_roi_grid_points.reshape(-1,3)
    # grid_points_color=grid_points
    # p3d.add_points(grid_points, grid_points_color)

    # p3d.show()
global_roi_grid_points = torch.load("./global_roi_grid_points['000008'].pt",map_location=torch.device('cpu') )
# print("new_xyz",new_xyz)
# print(new_xyz.size())
# new_xyz_batch_cnt = torch.load("./new_xyz_batch_cnt.pt",map_location=torch.device('cpu') )
# print("new_xyz_batch_cnt",new_xyz_batch_cnt)
np_global_roi_grid_points=global_roi_grid_points.cpu().detach().numpy()
a_file = open("data['000008'].pkl", "rb")
batch_dict = pickle.load(a_file)
print(str(batch_dict["frame_id"]))
spherical_project(batch_dict["frame_id"],batch_dict["rois"],np_global_roi_grid_points)