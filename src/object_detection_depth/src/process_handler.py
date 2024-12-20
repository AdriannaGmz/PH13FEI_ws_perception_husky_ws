#!/home/shared/venvs/python3_isa/bin/python3

import sys
import queue 
import rospy
import os
from transforms import *
import cv2
from cv_bridge import CvBridge, CvBridgeError
from retrieve_depth import DepthEstimator

import sensor_msgs.point_cloud2 as pc2
import message_filters
from sensor_msgs.msg import Image, PointCloud2

from object_detection_2d.msg import BoundingBoxes, BoundingBox2D
from geometry_msgs.msg import PoseWithCovariance as pwc
from geometry_msgs.msg import PoseArray, Pose
from object_detection_depth.msg import CustomPosewithCovarianceStamped as custom_msg

class HuskyDepth(object):
    def __init__(self):
        self.img_subscriber = None
        self.pc_subscriber = None
        self.preds_subscriber = None
        self.obj_depth_publisher = None
        self.q = queue.Queue()
        # TODO - checkout threading.Thread(target=self.processor, args=()).start()
        # self.processor()
        self.msg_seq = 0
        self.timestamp = None

    def processor(self,):
        
        while not rospy.is_shutdown():
            try:
                data_dict = self.q.get(block=False)
                
                rgb_img = CvBridge.imgmsg_to_cv2(data_dict['rgb_img'], 'passthrough')
                pc_osensor = get_os_data(data_dict['pc_osensor'])
                obj_bbox = get_preds(data_dict['obj_dets'])
                
                d_estimator = DepthEstimator(pc_osensor, obj_bbox, rgb_img)
                obj_list = d_estimator.main_husky()
                
                depth_msg = self.create_depth_msg(obj_list)

                self.obj_bev_publisher.publish(depth_msg)
                self.msg_seq += 1
            except queue.Empty:
                rospy.sleep(0.01)
 
    def create_depth_msg(self, obj_list, timestamp):
        msg = custom_msg()
        msg.header.seq = self.msg_seq
        msg.header.stamp = timestamp
        msg.header.frame_id = 'os_sensor'
        msg.pose = []
        for obj in obj_list.values():
            center = obj['lid_center']
            ext = obj['extents']
            cov = np.zeros(36, dtype=np.float)
            posecov = pwc()
            posecov.pose.position.x = center[0]
            posecov.pose.position.y = center[1]
            posecov.pose.position.z = center[2]
            cov[0] = abs(ext[0] - center[0])
            cov[7] = abs(ext[1] - center[1])
            cov[13] = abs(ext[5] - center[2])
            posecov.covariance = cov
            msg.pose.append(posecov)
        # rospy.loginfo(msg)
        return msg

    def create_pose_array(self, obj_list, timestamp):
        msg = PoseArray()
        msg.header.seq = self.msg_seq
        msg.header.stamp = timestamp
        msg.header.frame_id = 'os_sensor'
        msg.poses = []
        for obj in obj_list.values():
            center = obj['lid_center']
            ext = obj['extents']
            cov = np.zeros(36, dtype=np.float)
            pose = Pose()
            pose.position.x = center[0]
            pose.position.y = center[1]
            pose.position.z = center[2]
            msg.poses.append(pose)
        # rospy.loginfo(msg)
        return msg

    def create_image(self, image, in_rgb_img):
        msg = Image()
        msg.header.seq = self.msg_seq
        msg.header.stamp = in_rgb_img.header.stamp
        msg.header.frame_id = in_rgb_img.header.frame_id
        msg.height = in_rgb_img.height
        msg.width = in_rgb_img.width
        msg.is_bigendian = in_rgb_img.is_bigendian
        msg.encoding = in_rgb_img.encoding
        msg.step = in_rgb_img.step
        msg.data = image.astype(np.uint8).tobytes()
        return msg

    def messageCallback(self, in_rgb_img, pc_osensor, obj_dets):
        # self.q.put({'rgb_img': rgb_img, 'pc_osensor': pc_osensor, 'obj_dets': obj_dets})

        # data_dict = self.q.get(block=False)
        rgb_img = np.frombuffer(in_rgb_img.data, dtype=np.uint8).reshape(in_rgb_img.height, in_rgb_img.width, -1)
        # rgb_img = CvBridge.imgmsg_to_cv2(rgb_img, 'passthrough')
        pc_osensor = get_os_data(pc_osensor)
        obj_bbox = get_preds(obj_dets)

        d_estimator = DepthEstimator(pc_osensor, obj_bbox, rgb_img)
        obj_list, proj_pts_vis = d_estimator.main_husky()

        bev_msg = self.create_depth_msg(obj_list, timestamp=in_rgb_img.header.stamp)
        bev_vis_msg = self.create_pose_array(obj_list, timestamp=in_rgb_img.header.stamp)
        fv_proj_vis_msg = self.create_image(proj_pts_vis, in_rgb_img)

        self.obj_bev_publisher.publish(bev_msg)
        self.obj_bev_vis_publisher.publish(bev_vis_msg)
        self.fv_proj_vis_publisher.publish(fv_proj_vis_msg)
        self.msg_seq += 1
    
    def subscribe_topics(self):
        self.img_subscriber = message_filters.Subscriber('/camera_array/cam0Left/image_raw', Image)
        self.pc_subscriber = message_filters.Subscriber('/ouster/points', PointCloud2)
        self.preds_subscriber = message_filters.Subscriber('/perception/dets2d', BoundingBoxes)
        all_subs = message_filters.ApproximateTimeSynchronizer([self.img_subscriber, self.pc_subscriber, self.preds_subscriber], 100, 0.5, allow_headerless=True)
        all_subs.registerCallback(self.messageCallback)

    def publish_topics(self):
        rospy.loginfo('Publishing')
        self.obj_bev_publisher = rospy.Publisher('/perception/objects_bev', data_class=custom_msg, queue_size=10)
        self.obj_bev_vis_publisher = rospy.Publisher('/perception/objects_bev_vis', data_class=PoseArray, queue_size=10)
        self.fv_proj_vis_publisher = rospy.Publisher('/perception/fv_proj_vis', data_class=Image, queue_size=10)
        
    def initialize_node(self):
        self.subscribe_topics()
        self.publish_topics()
        rospy.loginfo('Depth Estimator Node Started')


def get_preds(obj_dets):
    obj_bbox = []
    for obj in obj_dets.bounding_boxes:
        obj_bbox.append([obj.id, obj.xmin, obj.ymin, obj.xmax, obj.ymax, obj.cat])
    return obj_bbox

def get_os_data(os_data):
    # self.timestamp = os_data.header.stamp
    pc_data = pc2.read_points(os_data, field_names=['x', 'y', 'z'], skip_nans=True)
    pc_data_np = np.array(list(pc_data))
    return pc_data_np
