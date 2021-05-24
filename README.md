# Rethink-FP-in-AVs-OpenPCDet

## Introduction 
* It will be an official code of our conference paper "Rethink the false positive objects in automotive vehicle". 
* The code is based on OpenPCDet under https://github.com/open-mmlab/OpenPCDet and https://nv-adlr.github.io/publication/2018-Segmentation
* This work will focus on solving the FP issue-> solo LiDAR method tends to detect lots of false positive objects.


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


## getting started
* download the semantic segmentation result from https://drive.google.com/drive/folders/1b98adm66H7gt3fOGzDv5s-uMYhSO9y2U?usp=sharing
* prepare your data as following
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & color_mask_2 & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools

## Changelog
* 24.05 edit OCCLUSION_DEPTH_MAP,ROI_GRID_POOL_ADD_MODULE in the config, edit kitti_dataset.py and pv_rcnn.py
  * save the depth map and semantic segmentation image in the kitti_dataset.py
  * calculate occlusion feature and semantic segmentation feature in the function _spherical_project_ in pv_rcnn.py
  * merge the pooled_features of grid points with the occlusion feature & semantic segmentation feature of grid point
  * several times of experiments showed that these module can effectively reduce the FP and get the stable improvement **+~5** for Ped and **+~3** for cyclist

## about Authors
* LeiChen Wang, PhD candidate at Uni Konstanz
* Simon Giebenhain, Master student at Uni Konstanz

