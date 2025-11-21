'''

TYRELL PROJECT

turtlebot3 Burger movement node

This node listens to the tb3_speed_time topic,
then it publishes in the cmd_vel topic 
the v and w speed values.
Those v and w values should be stored using the message type
Tb3 in the tyrell_interfaces.
(Any user defined message should be stored
in a file which begins with an uppercase)

Besides, it also publishes some lidar scan info
to the tb3_scan_sector.
Such scan info should be stored using the message type
Tb3ScanSector also included in the tyrell_interfaces.

The cmd_vel and scan topics are defined by the turtlebot,
so they are mandatory in case you need to work with that info.
However, tb3_speed_time and tbe_scan_sector topics
are defined by this node and they can be changed as needed.

The publication rate of speed values in cmd_vel topic
depends on the period of the timer.
After some real experiments, even with a 5 secs. period
the turtlebot keeps on moving and does not stop.

Since this node subscribes to the /scan topic, 
it has to define a proper QoS because that topic QoS
does not match the default subscriber QoS. More info:
- https://medium.com/@ultroninverse/mastering-ros-2-qos-profiles-a-practical-field-guide-on-reliability-latency-scalability-b3562eb70a26
- https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html#qos-profiles
If you need to check the QoS of a topic: ros2 topic info /scan --verbose

The timer period can be adapted to the requirements of the turtlebot topics
this node is subscribed to. In order to get the publishing rate of such topics,
you can use: ros2 topic hz tb3_scan_sector
Please notice that that frequency measure is not always accurate, since
a lot of different factors can affect to the publishing rate:
https://get-help.theconstruct.ai/t/ros2-hz-topic-increase-publishing-rate/21432

Remember to source install/local_setup.bash
after each colcon build!!

Ana Cruz-Martín
November 2025

'''


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from tyrell_interfaces.msg import Tb3
from tyrell_interfaces.msg import TbScanSector


class TyrellTb3Move(Node):

    def __init__(self):
        super().__init__('tyrell_tb3_move')
        
        # turtlebot3 linear and angular speed values
        self.v = 0.0
        self.w = 0.0
       
        # The sectors into we have split the lidar scan
        self.lidars1 = 0.0
        self.lidars2 = 0.0
        self.lidars3 = 0.0
        self.lidars4 = 0.0
             
        # publishers
        self.sensors_publisher = self.create_publisher(TbScanSector, 'tb3_scan_sector', 10)                                                                           
        self.speed_publisher = self.create_publisher(Twist, 'cmd_vel', 10)   
        
        #subscribers
        
        # QoS Profile required by the lidar subscriber    
        lidarQoS = QoSProfile(
           reliability=ReliabilityPolicy.BEST_EFFORT,
           durability=DurabilityPolicy.VOLATILE,
           history=HistoryPolicy.KEEP_LAST,
           depth=10)
           
        self.speed_time_sub = self.create_subscription( 
            Tb3,
            'tb3_speed_time',
            self.speed_time_callback,
            10)         
        self.lidar_sub = self.create_subscription( 
            LaserScan,
            '/scan',
            self.lidar_callback,
            lidarQoS)
        
        # Timer callback
        self.period = 0.5
        self.timer = self.create_timer(self.period, self.timer_callback)
        
        print('[INIT] tyrell_tb3_move node initialized', flush=True)
        
    def speed_time_callback(self, msg):
        # Maybe a safety-stop should be added to this callback,
        # just to assure that the turtlebot stops 
        # if there is no speed message in the tb3_speed_time topic
        self.v = msg.v
        self.w = msg.w
        self.period = msg.t
        print('[SPEED_TIME CALLBACK] Receiving: %f %f %f' % (self.v,self.w,self.period), flush=True) 
        
    def lidar_callback(self, msg): 
        # This callback should perform a more interesting lidar info processing than this :)
        # Such processing should be faster than the /scan publishing rate
        lidar_ranges = msg.ranges
        self.lidars1 = lidar_ranges[0]
        self.lidars2 = lidar_ranges[1]
        self.lidars3 = lidar_ranges[2]
        self.lidars4 = lidar_ranges[3]
        print('[LIDAR CALLBACK] Receiving: %f %f %f %f' % (self.lidars1,self.lidars2,self.lidars3,self.lidars4), flush=True)
        
    def timer_callback(self):
        speedMsg = Twist()
        speedMsg.linear.x = self.v
        speedMsg.angular.z = self.w
        self.speed_publisher.publish(speedMsg)
        print('[TIMER CALLBACK] Published speed info: %f %f' % (speedMsg.linear.x,speedMsg.angular.z), flush=True)
                    
        lidarMsg = TbScanSector()
        lidarMsg.s1 = self.lidars1
        lidarMsg.s2 = self.lidars2
        lidarMsg.s3 = self.lidars3
        lidarMsg.s4 = self.lidars4
        self.sensors_publisher.publish(lidarMsg)
        print('[TIMER CALLBACK] Published lidar info: %f %f %f %f' % (lidarMsg.s1,lidarMsg.s2,lidarMsg.s3,lidarMsg.s4), flush=True)
        

def main(args=None):
    rclpy.init(args=args)

    tyrell_tb3_move = TyrellTb3Move()
    print('[MAIN] Node created!!', flush=True)
    
    rclpy.spin(tyrell_tb3_move)
    print('[MAIN] Everybody is spinning!!', flush=True)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tyrell_tb3_move.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
