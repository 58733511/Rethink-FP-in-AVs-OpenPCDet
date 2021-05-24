#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:27:15 2021

@author: leichen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 13:56:22 2021

@author: leichen
"""

import copy
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.nuscenes import NuScenesExplorer
import matplotlib.pyplot as plt
import os.path as osp
from PIL import Image
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box

###########################
from typing import Tuple, List, Iterable
import os
import sys
# from generate_ground_truth import find_ghost_target
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_classes import Box


def find_ghost_target( points_radar, points_lidar,mask_GT
        ) -> Tuple: 
    points_ghost_radar= np.zeros_like(points_radar)
    
    for i in range (points_radar.shape[1]):
        np_min_dis= np.zeros((2,points_lidar.shape[1]))
        np_min_dis_x= np.zeros((2,points_lidar.shape[1]))
        np_min_dis_y= np.zeros((2,points_lidar.shape[1]))
        np_min_dis_x[0,:]= points_lidar[0,:]-points_radar[0,i]
        np_min_dis_y[1,:]= points_lidar[1,:]-points_radar[1,i]
        np_min_dis= np.sqrt(np_min_dis_x[0,:]*np_min_dis_x[0,:]+np_min_dis_y[1,:]*np_min_dis_y[1,:])
        # dist = np.linalg.norm(points_lidar[:2,:]-points_radar[:2,i])
        min_dis= np.amin(np_min_dis)
        if min_dis>1.5:
            points_ghost_radar[:,i]=points_radar[:,i]
            
    mask_lidar = np.ones(points_radar.shape[1], dtype=bool)
    mask_lidar = np.logical_and(mask_lidar, points_ghost_radar[0, :] !=0)
    mask_No_GT=np.logical_not(mask_GT)
    mask = np.logical_and(mask_lidar, mask_No_GT)
    points_ghost_radar = points_radar[:, mask_lidar]
    return points_ghost_radar,mask

def collect_point_label(sample_token, out_filename, file_format='txt'):
    my_sample = nusc.get('sample', sample_token)
    pointsensor_token = my_sample['data']["RADAR_FRONT"]
    camera_token = my_sample['data']["CAM_FRONT"]

    ########## show the image end
    max_sweeps=3
    sample_rec = nusc.get('sample', sample_token)
    ref_chan ="LIDAR_TOP"
    Point_F, times_F = RadarPointCloud.from_file_multisweep(nusc, sample_rec, "RADAR_FRONT", ref_chan, nsweeps=max_sweeps)
    pc_F_times = np.concatenate((Point_F.points, times_F))

    Point_F_L, times_F_L = RadarPointCloud.from_file_multisweep(nusc, sample_rec, "RADAR_FRONT_LEFT", ref_chan, nsweeps=max_sweeps)
    pc_F_L_times = np.concatenate((Point_F_L.points, times_F_L))

    Point_F_R, times_F_R = RadarPointCloud.from_file_multisweep(nusc, sample_rec, "RADAR_FRONT_RIGHT", ref_chan, nsweeps=max_sweeps)
    pc_F_R_times = np.concatenate((Point_F_R.points, times_F_R))

    Point_B_L, times_B_L = RadarPointCloud.from_file_multisweep(nusc, sample_rec, "RADAR_BACK_LEFT", ref_chan, nsweeps=max_sweeps)
    pc_B_L_times = np.concatenate((Point_B_L.points, times_B_L))
    Point_B_R, times_B_R = RadarPointCloud.from_file_multisweep(nusc, sample_rec, "RADAR_BACK_RIGHT", ref_chan, nsweeps=max_sweeps)
    pc_B_R_times = np.concatenate((Point_B_R.points, times_B_R))
    pc_times_radar = np.concatenate((pc_F_times, pc_F_L_times, pc_F_R_times, pc_B_L_times, pc_B_R_times), axis=1)  
    ref_chan ="LIDAR_TOP"
    Point_LiDAR, times_LiDAR = LidarPointCloud.from_file_multisweep(nusc, sample_rec, "LIDAR_TOP", ref_chan, nsweeps=max_sweeps)
    pc_times_lidar = np.concatenate((Point_LiDAR.points, times_LiDAR))
    
    #####################################################
    #####################################################transform block
    ####F
    LIDAR_TOKEN=my_sample["data"]["LIDAR_TOP"]

    RADAR_FRONT_TOKEN=my_sample["data"]["RADAR_FRONT"]

    pointsensor = nusc.get('sample_data', LIDAR_TOKEN)
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    Point_F.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    Point_F.translate(np.array(cs_record['translation']))
    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    Point_F.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    Point_F.translate(np.array(poserecord['translation']))
    
    #### FL
    RADAR_FRONT_LEFT_TOKEN=my_sample["data"]["RADAR_FRONT_LEFT"]
    # pointsensor = nusc.get('sample_data', RADAR_FRONT_LEFT_TOKEN)
    # cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    Point_F_L.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    Point_F_L.translate(np.array(cs_record['translation']))
    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    Point_F_L.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    Point_F_L.translate(np.array(poserecord['translation']))
    
    ###FR
    RADAR_FRONT_RIGHT_TOKEN=my_sample["data"]["RADAR_FRONT_RIGHT"]
    # pointsensor = nusc.get('sample_data', RADAR_FRONT_RIGHT_TOKEN)
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    Point_F_R.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    Point_F_R.translate(np.array(cs_record['translation']))
    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    Point_F_R.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    Point_F_R.translate(np.array(poserecord['translation']))
    
    ###BL
    RADAR_BACK_LEFT_TOKEN=my_sample["data"]["RADAR_BACK_LEFT"]
    # pointsensor = nusc.get('sample_data', RADAR_BACK_LEFT_TOKEN)
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    Point_B_L.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    Point_B_L.translate(np.array(cs_record['translation']))
    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    Point_B_L.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    Point_B_L.translate(np.array(poserecord['translation']))
    
    ###BR
    RADAR_BACK_RIGHT_TOKEN=my_sample["data"]["RADAR_BACK_RIGHT"]
    # pointsensor = nusc.get('sample_data', RADAR_BACK_RIGHT_TOKEN)
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    Point_B_R.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    Point_B_R.translate(np.array(cs_record['translation']))
    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    Point_B_R.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    Point_B_R.translate(np.array(poserecord['translation']))
    
    pc_radar_global = np.concatenate((Point_F.points, Point_F_L.points, Point_F_R.points, Point_B_L.points, Point_B_R.points), axis=1)  
    
    #####find the points in GT boxes!!

    lidar_front_data = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
    sample_data_token = lidar_front_data['token']


    boxes = nusc.get_boxes(sample_data_token)
    mask_GT = np.zeros(pc_radar_global.shape[1], dtype=bool)
    pc_radar_global[2,:]=1.0
    for box in boxes:
        mask_GT_box = points_in_box(box, pc_radar_global[:3,:], wlh_factor=1.5)
        mask_GT=np.logical_or(mask_GT_box, mask_GT)

    pc_times_radar_ghost,ghost_mask=find_ghost_target(pc_times_radar, pc_times_lidar,mask_GT)
    pc_times_radar[17,:]= 1

    pc_times_radar[17,ghost_mask]= 0
    if file_format=='numpy':
        np.save(out_filename, pc_times_radar)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
            (file_format))
        exit()

#########################
########################
#start
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

output_folder = os.path.join("/home/leichen/OpenPCDet/", 'data/nuscenes/')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


nusc = NuScenes(version='v1.0-mini', dataroot='/home/leichen/OpenPCDet/data/nuscenes/v1.0-mini', verbose=True)
nusc.list_scenes()

verbose=True
j=0
for i in range(len(nusc.scene)):
    my_scene = nusc.scene[i]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    while my_sample["next"] !="":
        j=j+1
        sample_token = my_sample["next"]
        my_sample = nusc.get('sample', sample_token)
        print("sample_token",sample_token)
        try:
            out_filename = sample_token+'Train_'+str(j)+'.npy' # Area_1_hallway_1.npy
            collect_point_label(sample_token, os.path.join(output_folder, out_filename), 'numpy')
        except:
            print(sample_token, 'ERROR!!')