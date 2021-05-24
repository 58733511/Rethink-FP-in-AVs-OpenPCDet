import torch.nn as nn
import torch
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from .roi_head_template import RoIHeadTemplate
import numpy as np
from ...utils import box_utils,calibration_kitti,common_utils
import cv2
import math

def get_seg_info_using_points_in_box(pts_rect,calib,bbox_lidar):
    corners_lidar = box_utils.boxes_to_corners_3d(bbox_lidar)
    flag = box_utils.in_hull(pts_rect[:, 0:3], corners_lidar)
    pts_in_box = pts_rect[flag,:]
    return pts_in_box 
       
class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]

        add_mlps = self.model_cfg.ROI_GRID_POOL_ADD_MODULE.MLPS
        for k in range(len(add_mlps)):
            add_mlps[k] = [4] + add_mlps[k]
            
        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )
        
        self.add_roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL_ADD_MODULE.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL_ADD_MODULE.NSAMPLE,
            mlps=add_mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL_ADD_MODULE.POOL_METHOD,
        )
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        
        #c_out = sum([x[-1] for x in mlps])
        #change
        c_out = sum([x[-1] for x in mlps])+sum([x[-1] for x in add_mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):

            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
        
    def spherical_project(self,batch_size,batch_dict,global_roi_grid_points):
        ############################################################################################
        ##### Created by LeiChen: include occlusion module and semantic segmentation image module
        ##### Occlusion module: proposed to get the occlusion info by calculating the depth difference 
        ##### between the depth of fake grid point and the real depth of the beam of laser (through spherical_projection).
        ##### For better understanding: imagine that a cyclist is occluded by a vehicle.
        ##### for a ROI of the target cyclist, we generate 6x6x6 fake grid points, whose depth is the depth 
        ##### of cyclist; and the the real depth of these laser beam should be the depth of vehicle.
        ############################################################################################
        ##### semantic segmentation image module: proposed to get the semantic segmentation info by 
        ##### get the semantic segmentation feature from the pre-train network
        ##### https://nv-adlr.github.io/publication/2018-Segmentation
        ############################################################################################

        ##### read parameter of depth map from config data 
        proj_H = self.model_cfg.OCCLUSION_DEPTH_MAP.OCCLUSION_DEPTH_MAP_HEIGHT
        proj_W = self.model_cfg.OCCLUSION_DEPTH_MAP.OCCLUSION_DEPTH_MAP_WIDTH
        proj_fov_up = self.model_cfg.OCCLUSION_DEPTH_MAP.OCCLUSION_DEPTH_MAP_FOV_UP
        proj_fov_down = self.model_cfg.OCCLUSION_DEPTH_MAP.OCCLUSION_DEPTH_MAP_FOV_DOWN
        fov_up = proj_fov_up / 180.0 * math.pi      # field of view up in rad
        fov_down = proj_fov_down / 180.0 * math.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
        cuda0 = global_roi_grid_points.device
        
        ##### get fake grid points
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)
        global_roi_grid_points_features=torch.zeros((global_roi_grid_points.shape[0],global_roi_grid_points.shape[1],4),dtype=torch.float32,device=cuda0)  # (B, Nx6x6x6, 1)
        for batch_num in range (batch_size):
            data_idx = batch_dict["frame_id"][batch_num]
            depth_plane = batch_dict["depth_plane"][batch_num]
            seg_feature = batch_dict["seg_feature"][batch_num]
            calib  = batch_dict["calib"][batch_num]
            global_roi_grid_points_each_batch = global_roi_grid_points[batch_num,:,:] # (1, Nx6x6x6, 3)
            global_roi_grid_points_each_batch_features=torch.zeros((global_roi_grid_points_each_batch.shape[0],4), dtype=torch.float32,device=cuda0)    # (Nx6x6x6, 1)
            
            ##### calculate the depth of fake grid points
            grid_points=global_roi_grid_points_each_batch.view(-1,3) #(Nx6x6x6, 3)
            depth_grid_points = torch.norm(grid_points, p=2, dim=1) #(Nx6x6x6, 1)
            
            ##### spherical project fake grid points
            grid_points_2d=torch.zeros((grid_points.shape[0],2),device=cuda0)
            grid_points_2d[:,0]=-torch.atan2(grid_points[:,1], grid_points[:,0])
            grid_points_2d[:,1]=torch.asin(grid_points[:,2] / depth_grid_points)
            grid_points_2d[:,0] = 0.5 * (grid_points_2d[:,0] / math.pi + 1.0)          # in [0.0, 1.0]
            grid_points_2d[:,1] = 1.0 - (grid_points_2d[:,1] + abs(fov_down)) / fov        # in [0.0, 1.0]
            grid_points_2d[:,0] *= proj_W                              # in [0.0, W]
            grid_points_2d[:,1] *= proj_H                              # in [0.0, H]
            grid_points_2d = torch.round(grid_points_2d).long()
            
            ##### filter the points, which are out of range of the depth map
            invalid_mask = (grid_points_2d[:,1] >= proj_H) | (grid_points_2d[:,1] < 0) | (grid_points_2d[:,0] >=proj_W) | (grid_points_2d[:,0] < 0)     
            grid_points_2d[invalid_mask,:] = 0

            ##### get the real depth of the beam of laser (through spherical_projection).
            real_depth = depth_plane[grid_points_2d[:,1],grid_points_2d[:,0],4] 
            
            ##### calculate the occlusion feature by "real_depth- depth_grid_points"
            global_roi_grid_points_each_batch_features[:,0] = real_depth- depth_grid_points
            ##### original version as backup
            '''
            occluded_depth_mask = depth_grid_points > (real_depth+3)
            global_roi_grid_points_each_batch_features[occluded_depth_mask,:]= 1 
            no_see_depth_mask = depth_grid_points < (real_depth-3)
            global_roi_grid_points_each_batch_features[no_see_depth_mask,:]= 0.333
            in_box_depth_mask = (depth_grid_points > (real_depth-3)) & (depth_grid_points < (real_depth+3))
            global_roi_grid_points_each_batch_features[in_box_depth_mask,:]= 0.667   
            '''
            ##### filter the points, which are out of range of the depth map
            global_roi_grid_points_each_batch_features[invalid_mask,0]= -100
            
            ############################################################################################
            ##### here start semantic segmentation image module
            ##### project fake grid point to the camera image
            grid_points = grid_points.cpu().detach().numpy()
            pts_in_Seg_Img = calib.lidar_to_img(grid_points[:,0:3])[0]
            
            ##### filter the points, which are out of range of the camera image
            pts_in_Seg_Img = pts_in_Seg_Img. astype(int)
            valid_mask = np.where((pts_in_Seg_Img[:,1] >=0) & (pts_in_Seg_Img[:,1] < seg_feature.shape[0])&
                                  (pts_in_Seg_Img[:,0] >=0) & (pts_in_Seg_Img[:,0] < seg_feature.shape[1]))
            pts_in_Seg_Img_copy = np.full_like(pts_in_Seg_Img,-100)
            pts_in_Seg_Img_copy[valid_mask] = pts_in_Seg_Img[valid_mask]
            
            ##### get the semantic segmentation feature of the projected grid points
            pts_seg_info = seg_feature[pts_in_Seg_Img_copy[:,1],pts_in_Seg_Img_copy[:,0]]
            pts_seg_info = torch.from_numpy(pts_seg_info).cuda()
            global_roi_grid_points_each_batch_features[:,1:4] = pts_seg_info[:,:3]
            global_roi_grid_points_features[batch_num,:,:] = global_roi_grid_points_each_batch_features
            
        global_roi_grid_points_features = global_roi_grid_points_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,4)  #(BxN, 6x6x6, 1)
        return global_roi_grid_points_features  
    
    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']
        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        
        global_roi_grid_points_features = self.spherical_project(batch_size,batch_dict,global_roi_grid_points) #(BxN, 6x6x6, 4)
        global_roi_grid_points_features = global_roi_grid_points_features.view(-1, 4)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)
        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)

        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)
        
        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)

        ############################################################################################
        ##### here add the module and features
        add_pooled_points, add_pooled_features = self.add_roi_grid_pool_layer(
            xyz=new_xyz,
            xyz_batch_cnt=new_xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=global_roi_grid_points_features.contiguous(),
        )  # (M1 + M2 ..., C)
        
        add_pooled_features = add_pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            add_pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        
        ##### concatenate the original feature of grid points and the added semantic & occlusion feature together
        pooled_features=torch.cat([pooled_features, add_pooled_features], dim=2) # (BxN, 6x6x6, C*2)
        
        return pooled_features 

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)
        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
