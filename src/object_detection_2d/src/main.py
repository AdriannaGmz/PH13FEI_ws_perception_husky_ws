#!/home/shared/venvs/python3_isa/bin/python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/shared/ws/perception_ws/src/")
sys.path.append("/home/shared/ws/perception_ws/src/object_detection_2d/")
import rospy
from handler import ObjectDetection2dHandler

if __name__ == "__main__":
    rospy.init_node("object_detection_2d", anonymous=True)
    od2d_handler = ObjectDetection2dHandler()
    od2d_handler.initializeNode()

    rospy.spin()