import os
from typing import Tuple
import numpy as np
from OpenGL.GL import glLineWidth
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numba
import argparse
import cv2
import pickle

def voxelize(
        points: np.ndarray,
        voxel_size: np.ndarray,
        grid_range: np.ndarray,
        max_points_in_voxel: int,
        max_num_voxels: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts 3D point cloud to a sparse voxel grid
    :param points: (num_points, num_features), first 3 elements must be <x>, <y>, <z>
    :param voxel_size: (3,) - <width>, <length>, <height>
    :param grid_range: (6,) - <min_x>, <min_y>, <min_z>, <max_x>, <max_y>, <max_z>
    :param max_points_in_voxel:
    :param max_num_voxels:
    :param include_relative_position: boolean flag, if True, the output num_features will include relative
    position of the point within the voxel
    :return: tuple (
        voxels (num_voxels, max_points_in_voxels, num_features),
        coordinates (num_voxels, 3),
        num_points_per_voxel (num_voxels,)
    )
    """
    points_copy = points.copy()
    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(np.int32)
    coor_to_voxelidx = np.full((grid_size[0], grid_size[1], grid_size[2]), -1, dtype=np.int32)
    voxels = np.zeros((max_num_voxels, max_points_in_voxel, points.shape[-1]), dtype=points_copy.dtype)
    coordinates = np.zeros((max_num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(max_num_voxels, dtype=np.int32)

    points_coords = np.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).astype(np.int32)
    mask = ((points_coords >= 0) & (points_coords < grid_size)).all(1)
    points_coords = points_coords[mask]
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == points_coords.shape[0]

    voxel_num = 0
    for i, coord in enumerate(points_coords):
        voxel_idx = coor_to_voxelidx[tuple(coord)]
        if voxel_idx == -1:
            voxel_idx = voxel_num
            if voxel_num > max_num_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[tuple(coord)] = voxel_idx
            coordinates[voxel_idx] = coord
        point_idx = num_points_per_voxel[voxel_idx]
        if point_idx < max_points_in_voxel:
            voxels[voxel_idx, point_idx] = points_copy[i]
            num_points_per_voxel[voxel_idx] += 1
    # know the posistion of voxel
    trans_coordinates=np.zeros_like(coordinates)
    trans_coordinates=trans_coordinates.astype(np.float32)
    for i in range (3):
        trans_coordinates[:,i]=coordinates[:,i]-((grid_range[i+3] - grid_range[i]) / (2*voxel_size[i]))
    return voxels[:voxel_num], coordinates[:voxel_num], num_points_per_voxel[:voxel_num],trans_coordinates[:voxel_num]

class Object3d(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        # location (x,y,z) in camera coord.
        self.t = (data[11], data[12], data[13])
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %d, %d, %f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %f, %f, %f, %f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %f, %f, %f" % (self.h, self.w, self.l))
        print(
            "!!!3d bbox location, ry: (%f, %f, %f), %f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )


# -----------------------------------------------------------------------------------------

def lidar_to_camera(points, r_rect, velo2cam):
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]

def project_to_image(points_3d, proj_mat):
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.zeros(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res

def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


class Calibration(object):
    """ Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs["P2"]
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs["R0_rect"]
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def read_calib_from_video(self, calib_root_dir):
        """ Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        """
        data = {}
        cam2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_cam_to_cam.txt")
        )
        velo2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_velo_to_cam.txt")
        )
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam["R"], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam["T"]
        data["Tr_velo_to_cam"] = np.reshape(Tr_velo_to_cam, [12])
        data["R0_rect"] = cam2cam["R_rect_00"]
        data["P2"] = cam2cam["P_rect_02"]
        return data

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        """ Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        """
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


# -----------------------------------------------------------------------------------------


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training"):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split

        self.lidar_dir = os.path.join(self.root_dir, self.split, "velodyne")
        self.label_dir = os.path.join(self.root_dir, self.split, "label_2")
        self.calib_dir = os.path.join(self.root_dir, self.split, "calib")

    def get_lidar(self, idx):
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        return load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return Calibration(calib_filename)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return read_label(label_filename)


# -----------------------------------------------------------------------------------------


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.
    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)
    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)
    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def compute_box_3d_no_obj(box_pred, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    obj.ry=box_pred[6]
    obj.l==box_pred[3]
    obj.w=box_pred[4]
    obj.h==box_pred[5]
    t_0=box_pred[0]
    t_1=box_pred[1]
    t_2=box_pred[2]
    R = roty(obj.ry)
    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] +t_0
    corners_3d[1, :] = corners_3d[1, :] +t_1
    corners_3d[2, :] = corners_3d[2, :] + t_2
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)
    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def compute_orientation_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)
    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])
    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]
    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)
    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P)
    return orientation_2d, np.transpose(orientation_3d)


# -----------------------------------------------------------------------------------------
def create_bbox_mesh_blue(p3d, gt_boxes3d):
    b = gt_boxes3d
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        p3d.add_line_blue([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]])
        i, j = k + 4, (k + 1) % 4 + 4
        p3d.add_line_blue([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]])
        i, j = k, k + 4
        p3d.add_line_blue([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]])



def create_bbox_mesh(p3d, gt_boxes3d):
    b = gt_boxes3d
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        p3d.add_line([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]])
        i, j = k + 4, (k + 1) % 4 + 4
        p3d.add_line([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]])
        i, j = k, k + 4
        p3d.add_line([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]])


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


def show_lidar_with_boxes(pc_velo, objects, calib):
    p3d = plot3d()
    points = pc_velo[:, 0:3]
    pc_color = pc_velo[:,4:]
    #print("pc_color",pc_color)
    p3d.add_points(points, pc_color)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        create_bbox_mesh(p3d, box3d_pts_3d_velo)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        p3d.add_line([x1, y1, z1], [x2, y2, z2])
    p3d.show()
    
def show_lidar_with_GT_Box_and_Pred_Box(pc_velo, objects, calib,pre_boxes):
    p3d = plot3d()
    points = pc_velo[:, 0:3]
    pc_color = inte_to_rgb(pc_velo[:,3])
    #print("pc_color",pc_color)
    p3d.add_points(points, pc_color)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        create_bbox_mesh(p3d, box3d_pts_3d_velo)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        p3d.add_line([x1, y1, z1], [x2, y2, z2])
    for pre_box in pre_boxes:
        obj.h = pre_box[3]  # box height
        obj.w = pre_box[5] # box width
        obj.l = pre_box[4] # box length (in meters)
        obj.t = (-pre_box[1], -pre_box[2], pre_box[0])
        obj.ry = pre_box[6]
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        create_bbox_mesh_blue(p3d, box3d_pts_3d_velo)
    p3d.show()
def line_to_point_min_distance( a, b, p):

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))
    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)


    # clamped parallel distance
    # h = np.maximum.reduce([s, t,0])
    h = np.maximum.reduce([s, t])
    mask = np.where(h<0)
    h[mask]=0

    # perpendicular distance component
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c,axis=1))


def show_lidar_with_voxels(pc_voxel,car_pc_voxel,num_in_voxel, objects, calib,pre_boxes):
    p3d = plot3d()
    # mask = num_in_voxel[:]<60
    # points = pc_voxel[mask, 0:3]
    points = car_pc_voxel

    mask_dist=np.sqrt(points[:,0]*points[:,0]+points[:,1]*points[:,1]+points[:,2]*points[:,2])>50   
    far_points = points[mask_dist, :]
    front_mask=far_points[:,0]>0

    # front_right_mask=(far_points[:,0]>0)&(far_points[:,1]>0)
    front_right_far_points=far_points[front_mask,:]
    ########generate fake voxel
    voxel_num=front_right_far_points.shape[0]
    times_fake=8
    fake_front_right_far_points=np.zeros((front_right_far_points.shape[0]*times_fake,3))
    for i in range (times_fake):
        fake_front_right_far_points[i*voxel_num:(i+1)*voxel_num,:]=front_right_far_points
    for i in range (times_fake):
        if i < 3:
            fake_front_right_far_points[i*voxel_num:(i+1)*voxel_num,0]=fake_front_right_far_points[i*voxel_num:(i+1)*voxel_num,0]+i
        elif i < 8:
            fake_front_right_far_points[i*voxel_num:(i+1)*voxel_num,1]=fake_front_right_far_points[i*voxel_num:(i+1)*voxel_num,1]+(i-5)

    ego_vehicle=np.array([0,0,4])
    for i in range (fake_front_right_far_points.shape[0]):
        if abs(np.min(line_to_point_min_distance(fake_front_right_far_points[i,:], ego_vehicle, pc_voxel))) > 0.5:
            fake_front_right_far_points[i,: ]=0
    # don't know why but z axis is low
    pc_voxel[:,2]=pc_voxel[:,2]-3

    pc_voxel[:,:3]=pc_voxel[:,:3].astype(np.float32)
    
    pc_voxel[:,:3]=pc_voxel[:,:3]/5
    p3d.add_points(pc_voxel, pg.glColor((255, 255, 0)),5)
    
    fake_front_right_far_points[:,2]=fake_front_right_far_points[:,2]-3

    fake_front_right_far_points[:,:3]=fake_front_right_far_points[:,:3].astype(np.float32)
    fake_front_right_far_points[:,:3]=fake_front_right_far_points[:,:3]/5
    p3d.add_points(fake_front_right_far_points, pg.glColor((0, 255, 0)),5)
    print("points x max", np.max(points[:, 0]))
    print("points x min", np.min(points[:, 0]))

    print("points y max", np.max(points[:, 1]))
    print("points y min", np.min(points[:, 1]))
    print("points z max", np.max(points[:, 2]))
    print("points z min", np.min(points[:, 2]))
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        create_bbox_mesh(p3d, box3d_pts_3d_velo)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        p3d.add_line([x1, y1, z1], [x2, y2, z2])
    for pre_box in pre_boxes:
        obj.h = pre_box[3]  # box height
        obj.w = pre_box[5] # box width
        obj.l = pre_box[4] # box length (in meters)
        obj.t = (-pre_box[1], -pre_box[2], pre_box[0])
        obj.ry = pre_box[6]
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        create_bbox_mesh_blue(p3d, box3d_pts_3d_velo)
    p3d.show()
    
    
    
def show_lidar_with_boxes_no_color(pc_velo, objects, calib):
    p3d = plot3d()
    points = pc_velo[:, 0:3]
    pc_inte = pc_velo[:, 3]
    p3d.add_points(points,"blue")
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        create_bbox_mesh(p3d, box3d_pts_3d_velo)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        p3d.add_line([x1, y1, z1], [x2, y2, z2])
    p3d.show()


def inte_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = 2 * (pc_inte - minimum) / (maximum - minimum)
    b = np.maximum((1 - ratio), 0)
    r = np.maximum((1 - ratio), 0)
    g = 1 - b
    return np.stack([r, g, b, np.ones_like(r)]).transpose()




def read_box_from_pkl(predictions,data_idx):
    for prediction in predictions:
        if "metadata" in prediction:
            if prediction["metadata"]["token"]==str(data_idx):
                #print(prediction["metadata"]["token"])
                box_pred=np.zeros((len(prediction["rotation_y"]),11), dtype=float)

                for i in range(len(prediction["rotation_y"])):
                    box_pred[i,0]=prediction["location"][i][2]
                    box_pred[i,1]=-prediction["location"][i][0]
                    box_pred[i,2]=-prediction["location"][i][1]
    
                    box_pred[i,3]=prediction["dimensions"][i][1]
                    box_pred[i,4]=prediction["dimensions"][i][0]
                    box_pred[i,5]=prediction["dimensions"][i][2]
    
                    box_pred[i,6]=prediction["rotation_y"][i]
                    
                return box_pred

        elif "frame_id" in prediction:
            
            if int(prediction["frame_id"])==data_idx:
                #print(prediction["metadata"]["token"])
                mask_car= np.where(prediction["name"]=="Car")
                box_pred_2d=prediction["bbox"][mask_car]

                prediction["rotation_y"]=prediction["rotation_y"][mask_car]
                prediction["location"]=prediction["location"][mask_car]
                prediction["dimensions"]=prediction["dimensions"][mask_car]
                prediction["bbox"]=prediction["bbox"][mask_car]

                box_pred=np.zeros((len(prediction["rotation_y"]),11), dtype=float)
    
                for i in range(len(prediction["rotation_y"])):
                    box_pred[i,0]=prediction["location"][i][2]
                    box_pred[i,1]=-prediction["location"][i][0]
                    box_pred[i,2]=-prediction["location"][i][1]
    
                    box_pred[i,3]=prediction["dimensions"][i][1]
                    box_pred[i,4]=prediction["dimensions"][i][0]
                    box_pred[i,5]=prediction["dimensions"][i][2]
    
                    box_pred[i,6]=prediction["rotation_y"][i]
                    box_pred[i,7]=prediction["bbox"][i][0]
                    box_pred[i,8]=prediction["bbox"][i][1]
                    box_pred[i,9]=prediction["bbox"][i][2]
                    box_pred[i,10]=prediction["bbox"][i][3]
                return box_pred

def show_result_with_camera_image(Camimg_filename,calib,bboxes,box_pred):
    Camimg = cv2.imread(Camimg_filename)
    for bbox in bboxes:
        print("bbox",bbox)
        points_1=(int(bbox[0]),int(bbox[1]))

        points_2=(int(bbox[0]),int(bbox[3]))
        points_3=(int(bbox[2]),int(bbox[3]))
        points_4=(int(bbox[2]),int(bbox[1]))
        Camimg = cv2.line(Camimg, points_1,points_2,(0,255,0), 1)
        Camimg = cv2.line(Camimg, points_2,points_3,(0,255,0), 1)
        Camimg = cv2.line(Camimg, points_3,points_4,(0,255,0), 1)
        Camimg = cv2.line(Camimg, points_4,points_1,(0,255,0), 1)

    for bbox in box_pred:
        points_1=(int(bbox[7]),int(bbox[8]))
        points_2=(int(bbox[7]),int(bbox[10]))
        points_3=(int(bbox[9]),int(bbox[10]))
        points_4=(int(bbox[9]),int(bbox[8]))
        Camimg = cv2.line(Camimg, points_1,points_2,(255,0,0), 1)
        Camimg = cv2.line(Camimg, points_2,points_3,(255,0,0), 1)
        Camimg = cv2.line(Camimg, points_3,points_4,(255,0,0), 1)
        Camimg = cv2.line(Camimg, points_4,points_1,(255,0,0), 1)
    cv2.imshow('Camimg',Camimg)
    

@numba.njit
def _add_rgb_to_points_kernel(points_2d, image, points_rgb):
    num_points = points_2d.shape[0]
    image_h, image_w = image.shape[:2]
    for i in range(num_points):
        img_pos = np.floor(points_2d[i]).astype(np.int32)
        if img_pos[0] >= 0 and img_pos[0] < image_w:
            if img_pos[1] >= 0 and img_pos[1] < image_h:
                points_rgb[i, :] = image[img_pos[1], img_pos[0], :]
                # image[img_pos[1], img_pos[0]] = 0


def add_rgb_to_points(points, image, rect, Trv2c, P2, mean_size=[5, 5]):
    kernel = np.ones(mean_size, np.float32) / np.prod(mean_size)
    # image = cv2.filter2D(image, -1, kernel)
    points_cam = lidar_to_camera(points[:, :3], rect, Trv2c)
    points_2d = project_to_image(points_cam, P2)
    points_rgb = np.zeros([points_cam.shape[0], 3], dtype=points.dtype)
    _add_rgb_to_points_kernel(points_2d, image, points_rgb)
    return points_rgb



if __name__ == "__main__":
    # at first give you dataset root and which scenes you want to check
    parser = argparse.ArgumentParser(description="KITTI LiDAR Viewer")
    parser.add_argument("--dataset-root", help="KITTI dataset root_dir")
    parser.add_argument("--num", type=int, help="Number of lidar samples to show")
    args = parser.parse_args()
    Camimg_dir = "/media/leichen/SeagateBackupPlusDrive/KITTI_DATABASE/training/image_2/"
    Calib_dir = "/media/leichen/SeagateBackupPlusDrive/KITTI_DATABASE/training/calib/"
    # read the prediction  
    with open('/home/leichen/results_det3d/pv_rcnn_result_editted.pkl', 'rb') as f:
        all_predictions = pickle.load(f)        
    dataset = kitti_object(args.dataset_root, "training")
    idxs = [
        int(i.split(".")[0])
        for i in os.listdir(os.path.join(args.dataset_root, "training", "velodyne"))
    ]
    
    for data_idx in range(0+args.num,7400):
        print(data_idx)
        Flag_useful=False
        objects = dataset.get_label_objects(data_idx)
        
        #select which data is relevant 
        #for example here I only check car
        for obj in objects:
            if (obj.type == "Car") and (int(obj.occlusion) >1):
                Flag_useful=True
            if (obj.type == "Car") and (float(obj.truncation) >=0.3) and (float(obj.truncation) <=0.5):
                Flag_useful=True
            obj.print_object()
        if Flag_useful!=True:
            continue
                #

        lidar_data = dataset.get_lidar(data_idx)
        calib = dataset.get_calibration(data_idx)
        # Show
        box=np.zeros((len(objects),7), dtype=float)
        box2d_GT=np.zeros((len(objects),4), dtype=float)
        i = 0
        
        for obj in objects:
            if obj.type != "Car":
                continue
            #obj.print_object()
            box[i,0]=obj.t[2]
            box[i,1]=-obj.t[0]
            box[i,2]=-obj.t[1]
            box[i,3]=obj.w
            box[i,4]=obj.l
            box[i,5]=obj.h
            box[i,6]=obj.ry
            box2d_GT[i,0]=obj.xmin
            box2d_GT[i,1]=obj.ymin
            box2d_GT[i,2]=obj.xmax
            box2d_GT[i,3]=obj.ymax
            i=i+1
            
            
        box_pred = read_box_from_pkl (all_predictions,data_idx)
        if  box_pred is None:
            continue
        Camimg_filename = os.path.join(Camimg_dir, "%06d.png" % (data_idx))
        print("Camimg_filename",Camimg_filename)
        show_result_with_camera_image(Camimg_filename,calib,box2d_GT,box_pred)
        calib_filepath = os.path.join(Calib_dir, "%06d.txt" % (data_idx))
        calib =Calibration(calib_filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        rect = calib.R0
        Trv2c = calib.V2C
        P2 = calib.P
        points = lidar_data
        show_lidar_with_GT_Box_and_Pred_Box(lidar_data,objects, calib,box_pred)