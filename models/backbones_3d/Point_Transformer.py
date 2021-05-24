import torch
import torch.nn as nn
from .pointnet_util import PointNetSetAbstraction
import torch.nn.functional as functional
import numpy as np
from scipy.interpolate import LinearNDInterpolator

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)




class TransitionDown(nn.Module):
    def __init__(self, npoint, nneighbor, channels) -> None:
        super().__init__()
        self.sa = PointNetSetAbstraction(npoint, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)
    
class TransitionUp(nn.Module):
    def __init__(self,npoint,channel) -> None:
        super().__init__()
        # P2_Fc means the FC on "downsampled input point set P2"-> linear + batch normal + ReLU
        self.P2_fc= nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        # P1_Fc means the FC on "higher resolution point set P1"
        self.P1_fc = nn.Linear(int(channel/2), int(channel/2), bias=False)
        
    def up_interpolation(new_xyz, xyz, points):
        inter_points = LinearNDInterpolator(xyz, points)
        new_points = inter_points(new_xyz)
        return new_points
    
    def forward(self, xyz_P2, points_P2, xyz_P1,points_P1):
        points_P2 = self.P2_fc(points_P2)
        new_points_P2 = self.up_interpolation(xyz_P1, xyz_P2, points_P2)
        new_points = new_points_P2+self.P1_fc(points_P1)
        return xyz_P1, new_points

    
class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, points: b x n x f
    def forward(self, xyz, points):


        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = points
        x = self.fc1(points)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = functional.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn
    
    
class PointTransformer(nn.Module):
    # fc_begin -> transformer_begin -> down_sample_0 -> down sample_1-> down sample_2-> down sample_3
    # -> fc_middle -> transformer_middle -> up_sample_0 -> up_sample_1-> up_sample_2 -> up_sample_3 -> fc_end
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        npoints=self.model_cfg.CONFIG.NPOINTS[0]
        nblocks=self.model_cfg.CONFIG.NBLOCKS[0]
        nneighbor=self.model_cfg.CONFIG.NNEIGHBOR[0]
        ### dim 3 -> xyz
        d_points=self.model_cfg.CONFIG.DPOINTS[0]
        d_transformer=self.model_cfg.CONFIG.DTRANSFORMER[0]
        # the first MLP 
        self.fc_begin = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        # the MLP in the middle, after down_sample_3

        self.fc_middle = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 32 * 2 ** nblocks),
            nn.ReLU(),
            nn.Linear(32 * 2 ** nblocks, 32 * 2 ** nblocks)
        )
        '''
        not sure how to do this-> to be finished
        self.fc_end = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        '''
        self.transformer_begin = TransformerBlock(32, d_transformer, nneighbor)
        self.transformer_middle= TransformerBlock(32 * 2 ** nblocks, d_transformer, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transition_ups = nn.ModuleList()
        self.transformer_downs = nn.ModuleList()
        self.transformer_ups = nn.ModuleList()

        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            print("channel",channel)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformer_downs.append(TransformerBlock(channel, d_transformer, nneighbor))
            
        for i in range(nblocks):
            channel = 32 * 2 ** (nblocks- i)
            print("channel",channel)
            self.transition_ups.append(TransitionUp(npoints // 4 ** (nblocks-i),channel))
            self.transformer_ups.append(TransformerBlock(channel, d_transformer, nneighbor))            

        self.nblocks = nblocks
        # not sure..
        self.num_point_features = self.model_cfg.CONFIG.DTRANSFORMER[0]

    # break the PC to xyz and features
    def break_up_pc(self, pc): 
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C) % what is vfe?
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor %?
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        input_points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(input_points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        ### from here
        ### xyz : only corrdinates of (sampled) points
        ### points : features of (sampled) points
        xyz = xyz.view(batch_size, -1, 3)
        print( "size",list(  xyz.size()  ) )
        xyz_begin = xyz
        points_begin = self.transformer_begin(xyz_begin, self.fc_begin(xyz_begin))[0]
        
        ### 4 transition_down + transformer_down block
        xyz_0, points_0 = self.transition_downs[0](xyz_begin, points_begin)
        points_0 = self.transformer_downs[0](xyz_0, points_0)[0]
        ### down sample_1 from N/4 to N/16
        xyz_1, points_1 = self.transition_downs[1](xyz_0, points_0)
        points_1 = self.transformer_downs[1](xyz_1, points_1)[0]
        ### down sample_2 from N/16 to N/64
        xyz_2, points_2 = self.transition_downs[2](xyz_1, points_1)
        points_2 = self.transformer_downs[2](xyz_2, points_2)[0]
        ### down sample_3 from N/64 to N/256
        xyz_3, points_3 = self.transition_downs[3](xyz_2, points_2)
        points_3 = self.transformer_downs[3](xyz_3, points_3)[0]      
        
        ### middle MLP
        points_middle = self.transformer_middle(xyz_3, self.fc_middle(xyz_3))[0]
        
        ### 4 transition_up + transformer_up block
        ### up sample_0 from N/256 to N/64
        xyz_4, points_4 = self.transition_ups[0](xyz_3, points_middle, xyz_2,points_2)
        points_4 = self.transformer_ups[0](xyz_4, points_4)[0]     
        ### up sample_1 from N/64 to N/16
        xyz_5, points_5 = self.transition_ups[1](xyz_4, points_4, xyz_1,points_1)
        points_5 = self.transformer_ups[1](xyz_5, points_5)[0]     
        ### up sample_2 from N/16 to N/4
        xyz_6, points_6 = self.transition_ups[2](xyz_5, points_5, xyz_0,points_0)
        points_6 = self.transformer_ups[2](xyz_6, points_6)[0]     
        ### up sample_3 from N/4 to N
        xyz_7, points_7 = self.transition_ups[3](xyz_6, points_6, xyz_begin,points_begin)
        points_7 = self.transformer_ups[3](xyz_7, points_7)[0]             
        
        batch_dict['point_features'] = points_7.view(-1, points_7.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), xyz_7[0].view(-1, 3)), dim=1)
        
        return batch_dict


        


