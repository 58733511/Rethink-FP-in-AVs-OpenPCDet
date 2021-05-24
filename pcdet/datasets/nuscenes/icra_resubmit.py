#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 19:34:06 2021

@author: leichen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:48:23 2021

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
import random

###########################
from typing import Tuple, List, Iterable

FLAG_FIGURE=True


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
    if FLAG_FIGURE:
        
        n_wrong_point = random.randint(1,15)
        for i in range(n_wrong_point):
            wrong_pos=random.randint(1,len(mask)-1)
            if mask[wrong_pos]==True:
                mask[wrong_pos]=False
            else:
                mask[wrong_pos]=True

    points_ghost_radar = points_radar[:, mask]
    return points_ghost_radar,mask



def map_pointcloud_to_image(pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0,
                                render_intensity: bool = False,
                                show_lidarseg: bool = False,
                                filter_lidarseg_labels: List = None,
                                lidarseg_preds_bin_path: str = None) -> Tuple: 
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        print("wrong")
    else:
        pc_0 = RadarPointCloud.from_file(pcl_path)
        #######leichen changed 
        pointsensor_next_token = pointsensor["next"]
        pointsensor_next = nusc.get('sample_data', pointsensor_next_token)
        pcl_path = osp.join(nusc.dataroot, pointsensor_next['filename'])
        pc_1 = RadarPointCloud.from_file(pcl_path)
        
        pointsensor_next_next_token = pointsensor_next["next"]
        pointsensor_next_next = nusc.get('sample_data', pointsensor_next_next_token)
        pcl_path = osp.join(nusc.dataroot, pointsensor_next_next['filename'])            
        pc_2 = RadarPointCloud.from_file(pcl_path)
        
        
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc_0.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc_0.translate(np.array(cs_record['translation']))
    ##### leichen changed!
    pc_1.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc_1.translate(np.array(cs_record['translation']))
    pc_2.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc_2.translate(np.array(cs_record['translation']))
    #####
    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc_0.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc_0.translate(np.array(poserecord['translation']))
    ##### leichen changed!
    pc_1.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc_1.translate(np.array(cs_record['translation']))
    pc_2.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc_2.translate(np.array(cs_record['translation']))
    #####
    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc_0.translate(-np.array(poserecord['translation']))
    pc_0.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
    ##### leichen changed!
    pc_1.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc_1.translate(np.array(cs_record['translation']))
    pc_2.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc_2.translate(np.array(cs_record['translation']))
    #####
    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc_0.translate(-np.array(cs_record['translation']))
    pc_0.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    ##### leichen changed!
    pc_1.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc_1.translate(np.array(cs_record['translation']))
    pc_2.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc_2.translate(np.array(cs_record['translation']))
    #####
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    radar_points = np.concatenate((pc_0.points, pc_1.points,pc_2.points),axis=1)

    depths = radar_points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(radar_points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]

    coloring = coloring[mask]

    return points, coloring, im

def get_color(category_name: str) -> Tuple[int, int, int]:
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """

    return nusc.colormap[category_name]

#nusc = NuScenes(version='v1.0-mini', dataroot='/media/leichen/SeagateBackupPlusDrive/NUSCNES/nuscenes_train', verbose=True)
nusc = NuScenes(version='v1.0-mini', dataroot='/home/leichen/OpenPCDet/data/nuscenes/v1.0-mini', verbose=True)

##### define
nusc.list_scenes()
my_scene = nusc.scene[7]
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
verbose=True
while my_sample["next"] !=None:
    sample_token = my_sample["next"]
    my_sample = nusc.get('sample', sample_token)
    print("sample_token",sample_token)
    max_sweeps=3
    sensor="radarlidar"
    # nusc.render_pointcloud_in_image(sample_token, pointsensor_channel='RADAR_FRONT',out_path="/home/leichen/Point Transformer/"+sample_token)
    
    ########## show the image

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = my_sample['data']["RADAR_FRONT"]
    camera_token = my_sample['data']["CAM_FRONT"]

    points, coloring, im = map_pointcloud_to_image(pointsensor_token, camera_token)
    ax=None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))

        fig.canvas.set_window_title(sample_token)
    else:  # Set title on if rendering as part of render_sample.
        ax.set_title("CAM_FRONT")
    ax.imshow(im)
    dot_size=15.0
    #ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    ax.axis('off')
    out_path="/home/leichen/Point Transformer/"+sample_token
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()
    ########## show the image end
 
  
    #####get the radar points
    if sensor=="radar":
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
        pc_times = np.concatenate((pc_F_times, pc_F_L_times, pc_F_R_times, pc_B_L_times, pc_B_R_times), axis=1)

    elif sensor== "lidar":
        sample_rec = nusc.get('sample', sample_token)
        ref_chan ="LIDAR_TOP"
        Point_LiDAR, times_LiDAR = LidarPointCloud.from_file_multisweep(nusc, sample_rec, "LIDAR_TOP", ref_chan, nsweeps=max_sweeps)
        pc_times = np.concatenate((Point_LiDAR.points, times_LiDAR))
    
    else:
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
    if sensor =="radar":
        radar_front_data = nusc.get('sample_data', sample_rec['data']["RADAR_FRONT"])
        sample_data_token = radar_front_data['token']
    elif sensor =="lidar":
        lidar_front_data = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        sample_data_token = lidar_front_data['token']
    else:
        lidar_front_data = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        sample_data_token = lidar_front_data['token']


        boxes = nusc.get_boxes(sample_data_token)
        mask_GT = np.zeros(pc_radar_global.shape[1], dtype=bool)
        pc_radar_global[2,:]=1.0
        for box in boxes:
            mask_GT_box = points_in_box(box, pc_radar_global[:3,:], wlh_factor=1.5)
            mask_GT=np.logical_or(mask_GT_box, mask_GT)


    use_flat_vehicle_coordinates=True
    ax=None
    underlay_map=True
    with_anns=True

    sd_record = nusc.get('sample_data', sample_data_token)
    
    sensor_modality = sd_record['sensor_modality']
    if sensor_modality in ['lidar', 'radar']:
        chan = sd_record['channel']
        ref_chan = 'LIDAR_TOP'
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_record = nusc.get('sample_data', ref_sd_token)
    
        if sensor == 'lidar':
    
            pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan,
                                                                 nsweeps=5)
            velocities = None
        elif sensor == "radar":        
            
        # Get aggregated radar point cloud in reference frame.
        # The point cloud is transformed to the reference frame for visualization purposes.
    
        # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
        # point cloud.
            radar_cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            ref_cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            velocities = pc_times[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc_times.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
            velocities[2, :] = np.zeros(pc_times.shape[1])
    
        else:
            pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan,
                                                                 nsweeps=5)
            
            radar_cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            ref_cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            velocities = pc_times_radar[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc_times_radar.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
            velocities[2, :] = np.zeros(pc_times_radar.shape[1])    # By default we render the sample_data top down in the sensor frame.
        # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
        # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
        if use_flat_vehicle_coordinates:
            # Retrieve transformation matrices for reference point cloud.
            cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                          rotation=Quaternion(cs_record["rotation"]))
    
            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        else:
            viewpoint = np.eye(4)
    
        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))
    
        # Render map if requested.
        if underlay_map:
            assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                 'otherwise the location does not correspond to the map!'
            axes_limit=100
            #NuScenesExplorer.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)
    
        # Show point cloud.
        
    
        
        if sensor== "radarlidar":
            
            points_lidar= view_points(pc_times_lidar[:3, :], viewpoint, normalize=False)
            dists_lidar = np.sqrt(np.sum(pc_times_lidar[:2, :] ** 2, axis=0))
    
            point_scale = 0.0005
            ax.scatter(points_lidar[0, :], points_lidar[1, :], c="gray", s=point_scale)
            
            points_radar= view_points(pc_times_radar[:3, :], viewpoint, normalize=False)
            dists_radar = np.sqrt(np.sum(pc_times_radar[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists_radar / axes_limit / np.sqrt(2))
            point_scale = 0.3
            
            ###### compare points_lidar & points_radar:
            points_radar_ghost,ghost_mask=find_ghost_target(points_radar, points_lidar,mask_GT)
            # scatter = ax.scatter(points_radar_ghost[0, :], points_radar_ghost[1, :], c="red", s=point_scale)
            dists_radar_ghost = np.sqrt(np.sum(points_radar_ghost[:2, :] ** 2, axis=0))
            colors=dists_radar_ghost
            scatter=ax.scatter(points_radar[0, :], points_radar[1, :], c="blue",s=point_scale)

        else :
            points= view_points(pc_times[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc_times[:2, :] ** 2, axis=0))
        
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
        
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
            
        if sensor == 'radar':
            points_vel = view_points(points[:3, :] + velocities, viewpoint, normalize=False)
            deltas_vel = points_vel - points
            deltas_vel = 6 * deltas_vel  # Arbitrary scaling
            max_delta = 20
            deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
            colors_rgba = scatter.to_rgba(colors)
            for i in range(points.shape[1]):
                ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])
                

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')
        box_vis_level= BoxVisibility.ANY
        # Get boxes in lidar frame.
        _, boxes, _ = nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)
    
        # Show boxes.
        if with_anns:
            for box in boxes:
                if "vehicle.car" in box.name:
                    print
                    c = np.array(get_color("vehicle.car")) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c,c,c),linewidth=1)
    
        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
    
    else:
        raise ValueError("Error: Unknown sensor modality!")
    
    ax.axis('off')
    
    ax.set_aspect('equal')
    out_path=None
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    verbose=True
    if verbose:
        plt.show()