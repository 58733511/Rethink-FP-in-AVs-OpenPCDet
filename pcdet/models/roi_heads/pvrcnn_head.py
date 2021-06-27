import torch.nn as nn
import torch
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from .roi_head_template import RoIHeadTemplate
import numpy as np
from ...utils import box_utils,calibration_kitti,common_utils
import cv2
import math
from .relation_3d import extract_position_matrix
###########################
#relation module
############################################################################################
##### Created by LeiChen
##### add an attention feature to FC feature, then feed to calculate cls_score/bbox_pred
##### reference: Relation-Networks-for-Object-Detection
##### https://nv-adlr.github.io/publication/2018-Segmentation
############################################################################################
class RelationModule(nn.Module):
    def __init__(self, n_relations = 16, appearance_feature_dim=1024,key_feature_dim = 64, geo_feature_dim = 64, isDuplication = False):
        super(RelationModule, self).__init__()
        self.isDuplication=isDuplication
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))
    def forward(self, input_data,position_embedding):
        if(self.isDuplication):
            f_a, embedding_f_a, position_embedding =input_data
        else:
            # suppose the input_data is only f_a
            f_a = input_data
        isFirst=True
        for N in range(self.Nr):
            if(isFirst):
                if(self.isDuplication):
                    concat = self.relation[N](embedding_f_a,position_embedding)
                else:
                    concat = self.relation[N](f_a,position_embedding)
                isFirst=False
            else:
                if(self.isDuplication):
                    concat = torch.cat((concat, self.relation[N](embedding_f_a, position_embedding)), -1)
                else:
                    concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
        concat = concat.view(concat.shape[0],concat.shape[1],1)

        return concat+f_a
    
class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=256,key_feature_dim = 64, geo_feature_dim = 64):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(256, self.dim_k, bias=True)
        self.WQ = nn.Linear(256, self.dim_k, bias=True)
        self.WV = nn.Linear(256, self.dim_k, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, f_a, position_embedding):
        N = f_a.shape[0]
        f_a = f_a.view(1,f_a.shape[0],f_a.shape[1])
        position_embedding = position_embedding.view(-1,self.dim_g)
        w_g = self.relu(self.WG(position_embedding))

        w_k = self.WK(f_a)
        w_k = w_k.view(N,1,self.dim_k)
        w_q = self.WQ(f_a)

        w_q = w_q.view(1,N,self.dim_k)

        w_v = self.WV(f_a)

        w_v = w_v.view(N,1,-1)
        
        scaled_dot = torch.sum((w_k*w_q),-1 )
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(N,N)

        w_a = scaled_dot.view(N,N)

        w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=1)(w_mn)
        

        output = w_mn*w_v

        output = torch.sum(output,-2)
        return output
###########################

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
        
        self.isDuplication=self.model_cfg.RELATION_MODULE.DUPLICATION
        self.Nr = self.model_cfg.RELATION_MODULE.N_RELATIONS
        self.dim_g = self.model_cfg.RELATION_MODULE.GEO_FEATURE_DIM
        self.fc_features = self.model_cfg.RELATION_MODULE.FC_FEATURE  
        self.dim_key = self.model_cfg.RELATION_MODULE.KEY_FEATURE  

        #c_out = sum([x[-1] for x in mlps])
        #change
        c_out = sum([x[-1] for x in mlps])+sum([x[-1] for x in add_mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        
        self.relation_module= RelationModule(n_relations = self.Nr, appearance_feature_dim=self.fc_features,
                                   key_feature_dim = self.dim_key, geo_feature_dim = self.dim_g)
        
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])          
                
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        # in order to put position embbeding after rule, so divide shared_fc_layer 
        # into shared_fc_layer_1 and shared_fc_layer_2
        #self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        self.shared_fc_layer_1 = nn.Sequential(shared_fc_list[0],shared_fc_list[1],shared_fc_list[2],shared_fc_list[3])
        self.shared_fc_layer_2 = nn.Sequential(shared_fc_list[4],shared_fc_list[5],shared_fc_list[6])

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

        # position embedding 
        position_embedding = extract_position_matrix (batch_dict['rois'])
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features_1 = self.shared_fc_layer_1(pooled_features.view(batch_size_rcnn, -1, 1))
        if self.model_cfg.RELATION_MODULE.FLAG:
            shared_features_1_plus_relation = self.relation_module (shared_features_1,position_embedding)
            shared_features_2 = self.shared_fc_layer_2(shared_features_1_plus_relation)
        else:
            shared_features_2 = self.shared_fc_layer_2(shared_features_1)            
        rcnn_cls = self.cls_layers(shared_features_2).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features_2).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

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
