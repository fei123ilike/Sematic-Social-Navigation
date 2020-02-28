#!/home/asus/torch_gpu_ros/bin/python
import rospy
import os
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose, Twist, PointStamped, Point
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import PointCloud2, Joy
import sensor_msgs.point_cloud2 as pc2
from collections import deque, defaultdict

class parseData(object):
    def __init__(self):
        self.active_peds_id = set()
        self.peds_pos_t = defaultdict(lambda: None)
        self.odomData = Pose()
        self.trackedPtSub = rospy.Subscriber("/tracked_points", PointCloud2, self.trackedPtCallback, queue_size=1)
        self.odomSub = rospy.Subscriber("/state_estimation", Odometry, self.odomCallback, queue_size=1)

    def write(self, filename, x, y, z):
        completeName = os.path.join("/home/asus/catkin_ws/src/navigan/txt/", filename +" .txt")
        with open(completeName, "a") as file:
            file.write(str(x) + ' ' + str(y) + ' '+ str(z) +'\n')
            rospy.loginfo('[{}]: write pose to {}'.format(rospy.get_name(), filename))
        return


    def odomCallback(self, _odomIn):
        odom_time = _odomIn.header.stamp

        self.odomData = _odomIn.pose.pose
        self.write("odom", self.odomData.position.x, self.odomData.position.y, self.odomData.position.z)
        rospy.loginfo('[{}]: odom received'.format(rospy.get_name()))

        return

    def trackedPtCallback(self, _scanIn):

        scanTime = _scanIn.header.stamp
        self.active_peds_id = set()
        self.peds_pos_t = defaultdict(lambda: None)

        for p in pc2.read_points(_scanIn, field_names = ("x", "y", "z", "h","s","v"), skip_nans=True):
            #active_peds_id.add(p[3])
            #peds_pos_t[p[3]] = np.array([p[0], p[1]])
            filename = "ped_"+str(int(p[3]))
            self.write(filename, p[0], p[1], p[2])
        rospy.loginfo('[{}]: tracked received'.format(rospy.get_name()))

if __name__ == '__main__':
    try:
        rospy.get_master().getPid()
    except:
        print("roscore is offline, exit")
        sys.exit(-1)


    rospy.init_node('parsing_dataset')
    writer = parseData()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('[{}] Shutting down...'.format(rospy.get_name()))
