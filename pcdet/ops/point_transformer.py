import torch
import torch.nn as nn
import torch.nn.functional as F
#from pointnet2.pointnet2_stack import pointnet2_utils

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
    #print(src.shape)
    #print(dst.shape)
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

class CrossTransformerBlock(nn.Module):
    def __init__(self, dim_inp_q, dim_inp_kv, dim, dim_out, skip_q=False):
        super().__init__()
        self.dim_inp_q = dim_inp_q
        self.dim_inp_kv = dim_inp_kv

        self.dim = dim
        #self.dim_out = dim_out

        self.fc_delta = nn.Sequential(
            nn.Linear(3, int(dim/2)),
            nn.ReLU(),
            nn.Linear(int(dim/2), dim)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        if not skip_q:
            self.w_qs = nn.Linear(dim_inp_q, dim, bias=False)
        self.w_ks = nn.Linear(dim_inp_kv, dim, bias=False)
        self.w_vs = nn.Linear(dim_inp_kv, dim, bias=False)

        self.fc_out = nn.Linear(dim, dim_out)




    def forward(self, xyz_q, feats_q, xyz_kv, feats_kv):
        """
        :param xyz_q: (N, 3) xyz coordinates of the keypoints (here called queries)
        :param feats_q: (N, dim_inp) features of queries
        :param xyz_kv: (N, k, 3) xyz coordinates of grid points
        :param feats_kv: (N, k, dim_inp) features of grid points
        :return:
            new_xyz: (N, 3) new xyz coordinates
            new_features: (N, dim_out) new_features for keypoints
        """
        #print(xyz_q.shape)
        #print(feats_kv.shape)

        if feats_q is not None:
            q_attn = self.w_qs(feats_q)  # (N, dim)
        k_attn = self.w_ks(feats_kv)  # (N, k, dim)
        v_attn = self.w_vs(feats_kv)  # (N, k, dim)
        pos_encode = self.fc_delta(xyz_q[:, None] - xyz_kv)  # (N, k, dim)

        if feats_q is not None:
            attn = self.fc_gamma(q_attn[:, None] - k_attn + pos_encode)
        else:
            attn = self.fc_gamma(pos_encode - k_attn)
        attn = F.softmax(attn, dim=-2)  # (N, k, dim)

        res = torch.einsum('nmf,nmf->nf', attn, v_attn + pos_encode)  # (N, dim)
        res = self.fc_out(res)

        return xyz_q, res


class TransformerBlock(nn.Module):
    def __init__(self, d_dict, k=None):#, k, radius=None, shift=False, group_all=False):
        super().__init__()

        self.d_model = d_dict['d_model']
        self.d_in = d_dict['d_in']
        self.d_in_coords = d_dict['d_in_coords']
        self.k = k

        ##if shift:
        #self.fc_offset = nn.Sequential(
        #    nn.Linear(self.d_in, self.d_model),
        #    nn.ReLU(),
        #    nn.Linear(self.d_model, self.d_in_coords)
        #)
        #self.rezero = nn.Parameter(torch.zeros(1))

        self.bn = nn.BatchNorm1d(self.d_in)
        #self.ln = nn.LayerNorm(self.d_in)

        self.fc_delta = nn.Sequential(
            nn.Linear(self.d_in_coords, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.w_qs = nn.Linear(self.d_in, self.d_model, bias=False)
        self.w_ks = nn.Linear(self.d_in, self.d_model, bias=False)
        self.w_vs = nn.Linear(self.d_in, self.d_model, bias=False)

        if not self.d_in == self.d_model:
            self.proj = nn.Linear(self.d_model, self.d_in)
        #self.k = k
        #self.radius = radius
        #self.shift = shift
        #self.group_all = group_all


    # xyz: b x n x d_coords, feats: b x n x d_in
    def forward(self, xyz, feats):

        b, n, _ = xyz.shape

        if self.k is None:
            knn_idx = torch.arange(n, device=xyz.device).unsqueeze(0).unsqueeze(1).repeat(b, n, 1)
        else:
            dists = square_distance(xyz, xyz)
            knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k

        knn_xyz = index_points(xyz, knn_idx)

        q_attn = self.w_qs(feats)
        k_attn = index_points(self.w_ks(feats), knn_idx)
        v_attn = index_points(self.w_vs(feats), knn_idx)

        pos_encode = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x n x f

        attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)

        attn = nn.functional.softmax(attn, dim=-2)  # b x n x n x f
        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode) # b x n x f
        if not self.d_in == self.d_model:
            res = self.proj(res)
        res = res + feats
        #res = self.ln(res)
        res = self.bn(res.permute(0, 2, 1)).permute(0, 2, 1)
        #xyz = xyz + self.rezero * self.fc_offset(res)
        return res #xyz, res

class TransformerBlockMHA(nn.Module):
    def __init__(self, d_dict, k=None, h=4):#, k, radius=None, shift=False, group_all=False):
        super().__init__()

        self.d_model = d_dict['d_model']
        self.d_in = d_dict['d_in']
        self.d_in_coords = d_dict['d_in_coords']
        self.k = k
        self.h = h

        #if shift:
        #    self.fc_offset = nn.Sequential(
        #        nn.Linear(self.d_model, self.d_model),
        #        nn.ReLU(),
        #        nn.Linear(self.d_model, self.d_in_coords)
        #    )
        #    self.rezero = nn.Parameter(torch.zeros(1))

        self.bn = nn.BatchNorm1d(self.d_in)

        self.fc_delta = nn.Sequential(
            nn.Linear(self.d_in_coords, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        self.w_qs = nn.Linear(self.d_in, self.d_model, bias=False)
        self.w_ks = nn.Linear(self.d_in, self.d_model, bias=False)
        self.w_vs = nn.Linear(self.d_in, self.d_model, bias=False)

        if not self.d_in == self.d_model:
            self.proj = nn.Linear(self.d_model, self.d_in)
        #self.k = k
        #self.radius = radius
        #self.shift = shift
        #self.group_all = group_all


    # xyz: b x n x d_coords, feats: b x n x d_in
    def forward(self, xyz, feats):

        b, n, _ = xyz.shape

        if self.k is None:
            knn_idx = torch.arange(n, device=xyz.device).unsqueeze(0).unsqueeze(1).repeat(b, n, 1)
        else:
            dists = square_distance(xyz, xyz)
            knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k

        knn_xyz = index_points(xyz, knn_idx)

        q_attn = self.w_qs(feats)
        k_attn = index_points(self.w_ks(feats), knn_idx)
        v_attn = index_points(self.w_vs(feats), knn_idx)

        pos_encode = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x n x f

        attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)

        attn = nn.functional.softmax(attn, dim=-2)  # b x n x n x f
        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode) # b x n x f
        if not self.d_in == self.d_model:
            res = self.proj(res)
        res = res + feats
        res = self.bn(res.permute(0, 2, 1)).permute(0, 2, 1)
        return res
