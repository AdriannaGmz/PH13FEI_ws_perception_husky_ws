#!/usr/bin/env python

# for every frame
# 1. load image, calib, label, velodyne'
# 2. project velodyne to image plane
# 3. get lidar points for each object
# 4. get bev image for each object
# 5. plot in x-y plane

import numpy as np
import transforms as transforms
import cv2
import matplotlib.pyplot as plt

def get_calib():
    K = np.eye(4)
    K[0, 0] = 252.01005633486346
    K[1, 1] = 251.61370309819347
    K[0, 2] = 256.7578657984281
    K[1, 2] = 195.86967490998143
    return K

class DepthEstimator(object):
    def __init__(self, pc_osensor, obj_boxes, rgb_img):
        
        self.pc_data = pc_osensor
        self.obj_boxes = obj_boxes
        self.rgb_img = rgb_img
        self.img_height, self.img_width, _ = rgb_img.shape
        
        self.factor = 0.9
        self.obj_list = {}

        self.pc_fov_pix = None
        self.pc_fov_depth = None
        self.pc_fov_data = None


    def get_obj_list(self):
        obj_list = {}
        for k, obj in enumerate(self.obj_boxes):

            # [obj.id, obj.xmin, obj.ymin, obj.xmax, obj.ymax, obj.cat]  <-- This is the order in the list
            xmin, ymin, xmax, ymax, cat = obj[1], obj[2], obj[3], obj[4], obj[5]

            center = [(xmin+xmax)/2, (ymin+ymax)/2]

            newxmin = (xmin - center[0]) * self.factor + center[0]
            newxmax = (xmax - center[0]) * self.factor + center[0]
            newymin = (ymin - center[1]) * self.factor + center[1]
            newymax = (ymax - center[1]) * self.factor + center[1]

            obj_list[k] = {'obj_cat': cat ,'box_coords' :[xmin, ymin, xmax, ymax], 'center': [(xmin+xmax)/2, (ymin+ymax)/2],
            'reduced_box': [newxmin, newymin, newxmax, newymax]}

        self.obj_list = obj_list

    def proj_pts(self):
        proj_mat = transforms.get_osensor_to_img(get_calib())
        pts_2d = transforms.proj_to_image(self.pc_data.transpose(), proj_mat)
        inds = np.where((pts_2d[0, :] < self.img_width) & (pts_2d[0, :] >= 0) &(pts_2d[1, :] < self.img_height) & (pts_2d[1, :] > 0) & (self.pc_data[:, 1] > 0))[0]
       
        self.pc_fov_pix = pts_2d[:, inds]
        self.pc_fov_data = self.pc_data[inds, :]
        self.pc_fov_depth = proj_mat @ np.hstack((self.pc_fov_data, np.ones((self.pc_fov_data.shape[0], 1)))).transpose()

    def obj_depth(self):
        for obj in self.obj_list.values():
            reduced_box = obj['reduced_box']
            index = np.where((self.pc_fov_pix[0, :] > reduced_box[0]) & (self.pc_fov_pix[0, :] < reduced_box[2]) & 
                        (self.pc_fov_pix[1, :] > reduced_box[1]) & (self.pc_fov_pix[1, :] < reduced_box[3]))[0]
           
            lid_points = self.pc_fov_data[index, :]

            obj['lid_indices'] = index if lid_points.shape[0] !=0 else []
            obj['avg_depth'] = np.mean(lid_points[:, 0]) if lid_points.shape[0] !=0 else 0
            obj['median_depth'] = np.median(lid_points[:, 0]) if lid_points.shape[0] !=0 else 0
        self.obj_extents()

    def obj_extents(self):
        for k, obj in self.obj_list.items():
            median = obj['median_depth']
            indices = obj['lid_indices']
            
            # In Lidar frame x - fwd, y - left, z - up
            if obj['obj_cat'] == 'Car':
                threshold = 4
            elif obj['obj_cat'] == 'Person':
                threshold = 1

            if len(indices) == 0:
                obj['lid_center'] = [0, 0, 0]
                obj['extents'] = [0, 0, 0, 0, 0, 0]
                continue
            
            obj_lid = self.pc_fov_data[indices, :]
            d_ind = np.where((obj_lid[:, 0] <= (median + threshold)) & (obj_lid[:, 0] >= (median -0.2)))[0]
            depth = obj_lid[d_ind, :]
            
            # print('depth_indices: ', depth.shape)
            if depth.shape[0] == 0:
                obj['lid_center'] = [0, 0, 0]
                obj['extents'] = [0, 0, 0, 0, 0, 0]
                continue 
            
            obj_xmin = np.min(depth[:,0])
            obj_xmax = np.max(depth[:,0])
            obj_ymin = np.min(depth[:,1])
            obj_ymax = np.max(depth[:,1])

            obj_zavg = np.mean(self.pc_fov_data[indices, 2])
            obj_zmax = obj_zavg
            obj_zmin = 0
            
            obj['lid_center'] = [(obj_xmax-obj_xmin)/2, (obj_ymax-obj_ymin)/2, obj_zavg/2]
            obj['extents'] = [obj_xmin, obj_ymin, obj_zmin, obj_xmax, obj_ymax, obj_zmax]

    def main_husky(self):
        self.get_obj_list()
        self.proj_pts()
        self.obj_depth()
        self.draw_img_proj()
        #self.plot = VisualizePlots(self.obj_list, self.pc_fov_pix, self.pc_fov_depth, self.rgb_img)

        return self.obj_list, self.rgb_img

    def draw_img_proj(self):
        cmap = plt.cm.get_cmap('hsv',)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        indices = self.pc_fov_pix.shape[1]
        for i in range(indices):
            cv2.circle(self.rgb_img, (int(np.round(self.pc_fov_pix[0, i])), int(np.round(self.pc_fov_pix[1, i]))), 1, color=(255, 0, 0), thickness=-1)
