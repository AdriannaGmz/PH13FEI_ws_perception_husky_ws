#!/usr/bin/env python3
import sys
sys.path.append('/home/shared/ws/perception_ws/src/object_detection_depth')
import rospy
from process_handler import HuskyDepth

if __name__ == '__main__':
    rospy.init_node('object_detection_depth', anonymous=True)
    h = HuskyDepth()
    h.initialize_node()
    rospy.spin()
