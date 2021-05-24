#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:38:52 2021

@author: leichen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:25:26 2021

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
import random

###########################
from typing import Tuple, List, Iterable

def sample_lidar (pc_times_lidar,pc_times_radar):
    print("pc_times_lidar.shape",pc_times_lidar.shape)
    print("pc_times_radar.shape",pc_times_radar.shape)

    sampled_lidar=np.zeros((pc_times_lidar.shape[0],0))
    key_point=np.zeros((2,pc_times_lidar.shape[1]))

    for i in range(pc_times_radar.shape[1]):
        key_point[0,:]=pc_times_radar[0, i]
        key_point[1,:]=pc_times_radar[1, i]
        dists_lidar = np.sqrt(np.sum((pc_times_lidar[:2, :]-key_point) ** 2, axis=0))

        lidar_index=dists_lidar.argsort()[:15][::-1]
        print("lidar_index",lidar_index)
        sampled_points = pc_times_lidar[:, lidar_index]
        sampled_lidar=np.append(sampled_lidar,sampled_points,axis=1)
    print("dists_lidar.shape",dists_lidar.shape)

    print("sampled_lidar.shape",sampled_lidar.shape)
    return sampled_lidar

    
    
    
    
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
    print("pc_0.points size",pc_0.points.shape)

    print("radar_points size",radar_points.shape)
    depths = radar_points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(radar_points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    print("points size",points.shape)

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
    print("points size",points.shape)

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
my_scene = nusc.scene[3]
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
verbose=True
while my_sample["next"] !=None:
    sample_token = my_sample["next"]
    my_sample = nusc.get('sample', sample_token)
    print("sample_token",sample_token)
    max_sweeps=3
    sensor="radar"
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
    ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
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
    #####show the results
    
    
    
    
    use_flat_vehicle_coordinates=True
    ax=None
    underlay_map=True
    with_anns=True
    if sensor =="radar":
        radar_front_data = nusc.get('sample_data', sample_rec['data']["RADAR_FRONT"])
        sample_data_token = radar_front_data['token']
    elif sensor =="lidar":
        lidar_front_data = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        sample_data_token = lidar_front_data['token']
    else:
        lidar_front_data = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        sample_data_token = lidar_front_data['token']
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
            axes_limit=65
            #NuScenesExplorer.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)
    
        # Show point cloud.
        
    
        
        if sensor== "radarlidar":
            sampled_pc_times_lidar=sample_lidar(pc_times_lidar,pc_times_radar)

            points_lidar= view_points(sampled_pc_times_lidar[:3, :], viewpoint, normalize=False)
            dists_lidar = np.sqrt(np.sum(sampled_pc_times_lidar[:2, :] ** 2, axis=0))
    
            point_scale = 1
            
            points_radar= view_points(pc_times_radar[:3, :], viewpoint, normalize=False)
            key_point=np.zeros((2,sampled_pc_times_lidar.shape[1]))
            key_point[0,:]=0
            key_point[1,:]=35
            dists_lidar = np.sqrt(np.sum((sampled_pc_times_lidar[:2, :]-key_point) ** 2, axis=0))
            colors = np.minimum(1, dists_lidar / 11/ np.sqrt(2))
            for i in range(len(colors)):
                if colors[i]< 0.2:
                    colors[i]=colors[i]+random.uniform(-0.15, 0.05)
                elif colors[i]< 0.5:
                    colors[i]=colors[i]+random.uniform(-0.35, 0.35)
                elif colors[i]< 0.6:
                    colors[i]=colors[i]+random.uniform(-0.4, 0.4)
                elif colors[i]< 0.8:
                    colors[i]=colors[i]+random.uniform(-0.6, 0.2)                    
            point_scale = 10
            print("points_lidar.shape",points_lidar.shape)
            ax.scatter(points_lidar[0, :], points_lidar[1, :], c= colors, s=point_scale)

            # scatter = ax.scatter(points_radar[0, :], points_radar[1, :], c=colors, s=point_scale)
            

            
        if sensor == 'radar':
            points_vel = view_points(points[:3, :] + velocities, viewpoint, normalize=False)
            deltas_vel = points_vel - points
            deltas_vel = 6 * deltas_vel  # Arbitrary scaling
            max_delta = 20
            deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
            # colors_rgba = scatter.to_rgba(colors)
            # for i in range(points.shape[1]):
                # ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])
                
        elif sensor == 'radarlidar':
            points_vel = view_points(pc_times_radar[:3, :] + velocities, viewpoint, normalize=False)
            deltas_vel = points_vel - points_radar
            deltas_vel = 6 * deltas_vel  # Arbitrary scaling
            max_delta = 20
            deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
            # colors_rgba = scatter.to_rgba(colors)
            # for i in range(points_radar.shape[1]):
                # ax.arrow(points_radar[0, i], points_radar[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])
    
        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')

    
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