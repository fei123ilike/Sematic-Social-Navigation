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

class generateTrajactory(object):
    def __init__(self):

        self.trackedPtPub = rospy.Publisher("/tracked_points", PointCloud2, queue_size=10)
        self.datapath = "/home/asus/SocialNavigation/src/navigan/datasets/zara1/test/"
        self.filename = "crowds_zara01.txt"
        self.data = []
        delim = '\t'
        with open(self.datapath + self.filename, "r") as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                self.data.append(line)
        self.data = np.asarray(self.data)
        self.frames_list = np.unique(self.data[:, 0]).tolist()
        self.frame_data = []
        for frame in self.frames_list:
            self.frame_data.append(self.data[frame == self.data[:, 0], :])
        self.counter = 0
        self.total_num_frame = len(self.frames_list)



    def publish(self):

        field_names = [PointField('x', 0, PointField.FLOAT32, 1),
                       PointField('y', 4, PointField.FLOAT32, 1),
                       PointField('z', 8, PointField.FLOAT32, 1),
                       PointField('h', 12, PointField.FLOAT32, 1),
                       PointField('s', 16, PointField.FLOAT32, 1),
                       PointField('v', 20, PointField.FLOAT32, 1)
                       ]

        x = np.array(self.frame_data[self.counter][:,2])
        y = np.array(self.frame_data[self.counter][:,3])
        z = np.zeros_like(x)
        #active_peds_id
        h = np.array(self.frame_data[self.counter][:,1])
        s = np.zeros_like(x)
        v = np.zeros_like(x)
        points = np.vstack((x,y,z,h,s,v)).T
        points = points.tolist()
        # print(points)
        header = Header()
        header.frame_id = "/map"
        pointCloud = pc2.create_cloud(header, field_names, points)
        pointCloud.header.stamp = rospy.Time.now()
        self.trackedPtPub.publish(pointCloud)






if __name__ == '__main__':
    rospy.init_node('trajactory_publisher')
    generator = generateTrajactory()
    try:
        rate = rospy.Rate(0.4)
        while not rospy.is_shutdown():
            if generator.counter < generator.total_num_frame:
                generator.publish()
                generator.counter += 1
                rate.sleep()
    except rospy.ROSInterruptException:
        rospy.loginfo('[{}] Shutting down...'.format(rospy.get_name()))
