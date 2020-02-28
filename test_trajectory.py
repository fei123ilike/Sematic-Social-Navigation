#!/home/asus/torch_gpu_ros/bin/python
import rospy
import os
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose, Twist, PointStamped, Point
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import PointCloud2, Joy, PointField
import sensor_msgs.point_cloud2 as pc2
from collections import deque, defaultdict

def listener():
    rospy.Subscriber("/tracked_points", PointCloud2, trackedPtCallback, queue_size=1)
    rospy.spin()

def trackedPtCallback(_scanIn):
    scanTime = _scanIn.header.stamp
    active_peds_id = set()
    peds_pos_t = defaultdict(lambda: None)
    for p in pc2.read_points(_scanIn, field_names = ("x", "y", "z", "h","s","v"), skip_nans=True):
        active_peds_id.add(p[3])
        peds_pos_t[p[3]] = np.array([p[0], p[1]])

    print(active_peds_id)
    print(peds_pos_t)
    return

if __name__ == '__main__':
    rospy.init_node('ped_trajectory', anonymous=True)
    listener()
