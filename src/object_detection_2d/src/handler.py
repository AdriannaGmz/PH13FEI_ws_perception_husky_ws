#!/home/shared/venvs/python3_isa/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import rospy
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2, numpy as np

from src.centernet.DLAnet import DlaNet
from src.centernet.predict import Predictor
from object_detection_2d.msg import BoundingBoxes, BoundingBox2D

class ObjectDetection2dHandler:
    def __init__(self):
        self.rgb_image_subscriber = None
        self.rgb_image_topic_name = None
        self.dets_2d_publisher = None
        self.dets_2d_vis_publisher = None
        self.dets_2d_topic_name = None
        self.dets_2d_vis_topic_name = None

        self.bridge = CvBridge()
        self.camera_config = None
        self.model = None

        self.use_gpu = True
        torch.cuda.empty_cache()

    def rgbImageCallback(self, in_rgb_img):
        rgb_img = np.frombuffer(in_rgb_img.data, dtype=np.uint8).reshape(in_rgb_img.height, in_rgb_img.width, -1)
        # try:
        #     rgb_img = self.bridge.imgmsg_to_cv2(in_rgb_img, 'passthrough')
        # except CvBridgeError as e:
        #     print(e)

        # Preprocess the image
        rgb_img = self.rgbPreProcess(rgb_img)

        # Send the image though the network and get the required detections
        dets_2d = self.regressBoundingBoxes(rgb_img)
        dets_2d_vis = self.drawBoundingBoxes(rgb_img, dets_2d)
        
        # Create the ROS messages from the generated data
        dets_2d_msg = self.createDets2dMessage(dets_2d, in_rgb_img)
        dets_2d_vis_msg = self.createDets2dVisMessage(dets_2d_vis, in_rgb_img)

        self.dets_2d_publisher.publish(dets_2d_msg)
        self.dets_2d_vis_publisher.publish(dets_2d_vis_msg)

    def rgbPreProcess(self, in_img):
        ''' Preprocess the image

            Args:
                image - the image that need to be preprocessed
            Return:
                images (tensor) - images have the shape (1，3，h，w)
        '''
        #rospy.loginfo('RGB Preprocess')
        # shrink the image size and normalize here
        in_image = (in_img / 255.).astype(np.float32)

        # from three to four dimension
        # (h, w, 3) -> (3, h, w) -> (1，3，h，w)
        in_img = in_image.transpose(2, 0, 1)
        in_img = torch.from_numpy(in_img)

        return in_img

    def regressBoundingBoxes(self, rgb_img):
        rgb_img = rgb_img.to(self.cuda_device)
        output, dets = self.predictor.process(rgb_img, self.model)

        # Process the detections and get the final detections
        dets_np = dets.detach().cpu().numpy()[0]

        # select detections above threshold
        threshold_mask = (dets_np[:, -2] > self.predictor.thresh_)  # class in -1
        dets_np = dets_np[threshold_mask, :]

        # Convert the detections to the original scale
        dets_original = self.predictor.input2image(dets_np, dataset='ISA')
        return dets_original

    def drawBoundingBoxes(self, rgb_img, dets_2d):
        rgb_img = rgb_img.permute(1, 2, 0).cpu().numpy() * 255
        for i in range(dets_2d.shape[0]):
            if int(dets_2d[i, 5] == 0):
                cv2.rectangle(rgb_img, \
                              (int(dets_2d[i, 0]), int(dets_2d[i, 1])), \
                              (int(dets_2d[i, 2]), int(dets_2d[i, 3])), \
                              (0, 255, 0), 1)
                cv2.putText(rgb_img, 'Car', (int(dets_2d[i, 0]), int(dets_2d[i, 1]) - 5), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            else:
                cv2.rectangle(rgb_img, \
                              (int(dets_2d[i, 0]), int(dets_2d[i, 1])), \
                              (int(dets_2d[i, 2]), int(dets_2d[i, 3])), \
                              (255, 0, 0), 1)
                cv2.putText(rgb_img, 'Person', (int(dets_2d[i, 0]), int(dets_2d[i, 1]) - 5), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        return rgb_img

    def createDets2dMessage(self, dets_2d, in_rgb_img):
        
        # Order is xmin, ymin, xmax, ymax
        cat = ['Car', 'Person']
        dets2d_msg = BoundingBoxes()
        dets2d_msg.header = in_rgb_img.header
        dets2d_msg.bounding_boxes = []
        #rospy.loginfo(dets_2d)
        #detections - (N, 6) - bbox values, score, category
        
        for i in range(dets_2d.shape[0]):
            bbox = BoundingBox2D()
            bbox.xmin = int(dets_2d[i, 0])
            bbox.ymin = int(dets_2d[i, 1])
            bbox.xmax = int(dets_2d[i, 2])
            bbox.ymax = int(dets_2d[i, 3])
            bbox.score = int(dets_2d[i, 4])
            bbox.cat = cat[int(dets_2d[i, 5])]
            dets2d_msg.bounding_boxes.append(bbox)

        #rospy.loginfo(' Bounding Boxes published ')
        return dets2d_msg
        
    def createDets2dVisMessage(self, dets_2d_vis, in_rgb_img):
        dets2d_vis_msg = Image()  # self.bridge.cv2_to_imgmsg(dets_2d_vis.astype(np.uint8), '8UC3')
        dets2d_vis_msg.header = in_rgb_img.header
        dets2d_vis_msg.height = in_rgb_img.height
        dets2d_vis_msg.width = in_rgb_img.width
        dets2d_vis_msg.is_bigendian = in_rgb_img.is_bigendian
        dets2d_vis_msg.encoding = in_rgb_img.encoding
        dets2d_vis_msg.step = in_rgb_img.step
        dets2d_vis_msg.data = dets_2d_vis.astype(np.uint8).tobytes()

        return dets2d_vis_msg

    def initializeNode(self):
        self.loadParameters()
        self.subscribeTopics()
        self.publishTopics()

    def publishTopics(self):
        # pass
        self.dets_2d_publisher = rospy.Publisher(name=self.dets_2d_topic_name, data_class=BoundingBoxes, queue_size=10)
        self.dets_2d_vis_publisher = rospy.Publisher(name=self.dets_2d_vis_topic_name,  data_class=Image, queue_size=10)

    def subscribeTopics(self):
        self.rgb_image_subscriber = rospy.Subscriber(name='/camera_array/cam0Left/image_raw', data_class=Image,
                                                     callback=self.rgbImageCallback)

    def loadParameters(self):
        self.rgb_img_topic_name = '/camera_array/cam0Left/image_raw/compressed'
        self.dets_2d_topic_name = '/perception/dets2d'
        self.dets_2d_vis_topic_name = '/perception/dets2d_vis'

        def loadNeuralNetwork():
            self.cuda_device = torch.device('cuda')
            self.model = DlaNet(34)
            self.model.load_state_dict(torch.load('/home/shared/ws/perception_ws/src/object_detection_2d/src/saved_model/isa_best.pth'))
            self.model.to(self.cuda_device)
            self.model.eval()

            self.predictor = Predictor(True, 'ISA')

        loadNeuralNetwork()

        rospy.loginfo('Perception | ObjectDetection2d Network successfully loaded! Ready to receive data')
