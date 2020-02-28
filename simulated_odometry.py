#!/home/asus/torch_gpu_ros/bin/python
import numpy as np
import math
from math import sin, cos, pi
import rospy
import tf
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped, PointStamped


rospy.init_node('odometry_publisher')
waypoint = PointStamped()
# callback function to sbubscribe path
def waypointCallBack(data):

    waypoint.header.frame_id = "/map"
    waypoint.header.stamp = data.header.stamp
    waypoint.point = data.point
    # waypoint.x = data.x
    # waypoint.y = data.y
    # waypoint.z = data.z

    return waypoint


odom_pub = rospy.Publisher("/state_estimation", Odometry, queue_size=10)
waypoint_sub = rospy.Subscriber("/way_point", PointStamped, waypointCallBack,queue_size=5)
odom_broadcaster = tf.TransformBroadcaster()


x_list = np.arange(0.0, 3.0, 0.1).tolist()
y_list = np.arange(0.0, 3.0, 0.1).tolist()
th = 0.785398 # 45 degree
vx = 0
vy = 0
vth = 0

current_time = rospy.Time.now()
last_time = rospy.Time.now()
step_before_predict  = 0
step_after_predict = 0
batch_start  = True
batch_count = 0

rate = rospy.Rate(0.4)
while not rospy.is_shutdown():
    current_time = rospy.Time.now()
    if (waypoint_sub.get_num_connections() == 0 or step_before_predict < 5):
        x = x_list[step_before_predict]
        y = y_list[step_before_predict]
        step_before_predict += 1
        rospy.loginfo("Step before prediction %s ", step_before_predict)
    else:
        x = waypoint.point.x
        y = waypoint.point.y
    rospy.loginfo("Simulated Odometry %s %s", x, y)


    odom_quat = tf.transformations.quaternion_from_euler(0, 0, th)

     # first, we'll publish the transform over tf
    odom_broadcaster.sendTransform(
         (x, y, 0.),
         odom_quat,
         current_time,
         "base_footprint",
         "odom"
         )

     # next, we'll publish the odometry message over ROS
    odom = Odometry()
    odom.header.stamp = current_time
    odom.header.frame_id = "/map"

     # set the position
    odom.pose.pose = Pose(Point(x, y, 0.), Quaternion(*odom_quat))

     # set the velocity
    odom.child_frame_id = "base_footprint"
    odom.twist.twist = Twist(Vector3(vx, vy, 0), Vector3(0, 0, vth))

     # publish the message
    odom_pub.publish(odom)

    last_time = current_time
    rate.sleep()
