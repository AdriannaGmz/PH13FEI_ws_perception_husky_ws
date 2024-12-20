#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_matrix(translation, rotation):
    matrix = np.eye(4)
    c_x = np.cos(rotation['r_x'])
    s_x = np.sin(rotation['r_x'])
    c_y = np.cos(rotation['r_y'])
    s_y = np.sin(rotation['r_y'])
    c_z = np.cos(rotation['r_z'])
    s_z = np.sin(rotation['r_z'])
    
    # x, y, z rot matrices
    x_mat = np.array([[1, 0, 0],[0, c_x, -s_x],[0, s_x, c_x]])
    y_mat = np.array([[c_y, 0, s_y],[0, 1, 0],[-s_y, 0, c_y]])
    z_mat = np.array([[c_z, -s_z, 0],[s_z, c_z, 0],[0, 0, 1]])
    
    rot_mat = z_mat @ y_mat @ x_mat  

    matrix[:3, :3] = rot_mat
    matrix[:3, 3] = np.array([translation['x'], translation['y'], translation['z']])
    
    return matrix

def get_matrix_quat(translation, quaternion):
    
    matrix = np.eye(4)
    rot_mat = R.from_quat([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']]).as_matrix()
    matrix[:3, :3] = rot_mat
    matrix[:3, 3] = np.array([translation['x'], translation['y'], translation['z']])
    
    return matrix

def base_cam_mat():
    translation = { 'x' : 0.260, 'y' : 0.20, 'z' : 0.650}
    rotation = { 'r_x' : -1.571, 'r_y' : 0, 'r_z' : -1.571}
    #quaternion = { 'x' : -0.499, 'y' : 0.50, 'z' : -0.499, 'w' : 0.499}
    return get_matrix(translation, rotation)
    #return get_matrix_quat(translation, quaternion)

def cam_base_mat():
    inv = np.linalg.inv(base_cam_mat())
    return inv

def base_osensor_mat():
    translation = { 'x' :-0.232, 'y' : 0.138, 'z' : 1.057}
    rotation = { 'r_x' : 0, 'r_y' : 0, 'r_z' : -1.571}
    #quaternion = { 'x' : 0.0, 'y' : 0.0, 'z' : -0.707, 'w' : 0.707}
    return get_matrix(translation, rotation)

def osensor_base_mat():
    inv = np.linalg.inv(base_osensor_mat())
    return inv

def lidar_osensor_mat():
    inv = np.linalg.inv(osensor_lidar_mat())
    return inv

def osensor_lidar_mat():
    translation = { 'x' : 0, 'y' : 0, 'z' : 0.036}
    quaternion = { 'x' : 0.0, 'y' : 0.0, 'z' : 1.0, 'w' : 0.0}
    return get_matrix_quat(translation, quaternion)

def get_osensor_to_img(K):
   # lid_osensor_mat = lidar_osensor_mat()
    osensor_base = osensor_base_mat()
    base_cam = base_cam_mat()
    T = np.array([[0.99956263, -0.02680943, -0.01248192, -0.22648084],
                  [-0.01351595, -0.03874109, -0.99915787, -0.43903306],
                  [0.02630329, 0.99888957, -0.0390865, -0.55559062],
                  [0, 0, 0, 1]])
    #T = np.array([[0.985, 0.173, 0.0, 0], [0.0 ,0, 1.0,  0], [0.173, -0.985, 0, 0], [0, 0, 0, 1]])
    # proj_mat = K @ base_cam @ osensor_base 
    proj_mat = K @ T
    assert proj_mat.shape == (4, 4), 'proj_mat shape is not 4x4'

    return proj_mat

def proj_to_image(points, proj_mat):
    num_pts = points.shape[1]
    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]

def read_calib_file(filepath):

    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def get_distortion_ceoff():
    distortion_coeffs = { 'k1':-0.19055686474229397,  'k2':0.05586346249249971, 
                         't1':-8.031519283962964e-06, 't2': 0.00028201931934843626}
    return distortion_coeffs
