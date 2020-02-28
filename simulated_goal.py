#!/usr/bin/env python
"""
Simple node to publish PointStamped messages at a random location
in order to practice working with tf.

Messages are published to the /navigan_goal topic.

Author: Felix Lu
"""

import sys
import rospy
from geometry_msgs.msg import PointStamped
# import tf2
import random
import numpy as np

def run():
    """ Initialize the goal publisher node."""

    point_pub = rospy.Publisher('/navigan_goal', PointStamped, queue_size=10)
    rospy.init_node('goal_publisher')
    rate = rospy.Rate(0.4)

    # Generate a random goal:
    goal = np.zeros(3)
    # goal[0:2] = np.round(np.random.rand(2) * 20 - 2, 1)
    goal[0] = 8.5
    goal[1] = 8.5
    # Create the point message.
    point = PointStamped()

    # Publish the goal at 2.5 HZ
    while not rospy.is_shutdown():
        point.header.stamp = rospy.Time.now()
        point.header.frame_id = "/map"
        point.point.x = goal[0]
        point.point.y = goal[1]
        point_pub.publish(point)
        rospy.loginfo("Simulated Goal %s %s", goal[0], goal[1])
        # print('Simulated Goal:',goal)
        rate.sleep()

if __name__ == '__main__':

    try:
        rospy.get_master().getPid()
    except:
        print("roscore is offline, exit")
        sys.exit(-1)

    try:
        run()
    except rospy.ROSInterruptException:
        pass
