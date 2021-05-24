# Rethink-FP-in-AVs-OpenPCDet

## Introduction 
* It will an official code of our conference paper "Rethink the false positive objects in automotive vehicle". 
* This work will focus on solving the issue-> solo LiDAR method tends to detect lots of false positive objects.


## Plan 
* Mai
  * upload "semantic segmentation image" module, "occlusion" module
  * upload "point transformer" module
  * train **PartA2** with our module

* Juni
  * optimize and upload "multi-modal attention" module
  * compare and summary the detection result between original **PartA2**/**PointRCNN**/**PVRCNN** and our version
  * submit result to KITTI benchmark
  
* Juli
  * write paper
  * clean code 


## Changelog
* 24.05 edit OCCLUSION_DEPTH_MAP,ROI_GRID_POOL_ADD_MODULE in the config, edit kitti_dataset.py and pv_rcnn.py
  * save the depth map and semantic segmentation image in the kitti_dataset.py
  * calculate occlusion feature and semantic segmentation feature in the function _spherical_project_ in pv_rcnn.py
  * merge the pooled_features of grid points and occlusion feature and semantic segmentation feature of grid point
  * several times of experiments show that these module can effectively reduce the FP and get the stable improvement **+~5** for Ped and **+~3** for cyclist
