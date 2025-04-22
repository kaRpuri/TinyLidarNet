#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import serialize_message
import message_filters

from sensor_msgs.msg import Joy, LaserScan, Imu
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('sim_data_collector')

        # -------------------- writer --------------------
        self.msg_counter = 0
        self.init_bag_writer()

        # -------------------- joystick gate -------------
        self.joy_sub         = self.create_subscription(Joy, '/joy',
                                                        self.joy_callback, 10)
        self.last_cmd_time   = None          # rclpy.clock.Time of last *active* /joy
        self.record_timeout  = 0.25          # seconds to keep recording after last cmd
        self.joy_deadband    = 0.05          # ignore tiny stick noise

        # -------------------- QoS & subscribers ---------
        sensor_qos = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.odom_sub     = message_filters.Subscriber(self, Odometry,
                                                       '/pf/pose/odom')
        self.lidar_sub    = message_filters.Subscriber(self, LaserScan,
                                                       '/scan', qos_profile=sensor_qos)
        self.imu_sub      = message_filters.Subscriber(self, Imu,
                                                       '/sensors/imu/raw')
        self.position_sub = message_filters.Subscriber(self, PoseStamped,
                                                       '/pf/viz/inferred_pose')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_sub,
             self.imu_sub, self.position_sub],
            queue_size=20, slop=0.05, allow_headerless=True)
        self.ts.registerCallback(self.sensor_callback)

        self.get_logger().info('Data collector node initialised.')

    # ---------- joystick callback -------------------------------------------------
    def joy_callback(self, msg: Joy):
        """Record time only if the message represents a real command
        (any button pressed or any axis beyond dead‑band)."""
        active_axes = any(abs(a) > self.joy_deadband for a in msg.axes)
        active_btns = any(msg.buttons)
        if active_axes or active_btns:
            self.last_cmd_time = self.get_clock().now()

    # ---------- rosbag writer setup ----------------------------------------------
    def init_bag_writer(self):
        storage_options  = StorageOptions(uri='sim_Dataset/test_levine2',
                                          storage_id='sqlite3')
        converter_options = ConverterOptions('', '')

        self.writer = SequentialWriter()
        self.writer.open(storage_options, converter_options)

        for topic in (
            TopicMetadata(name='odom', type='nav_msgs/msg/Odometry',
                          serialization_format='cdr'),
            TopicMetadata(name='scan', type='sensor_msgs/msg/LaserScan',
                          serialization_format='cdr'),
            TopicMetadata(name='pose', type='geometry_msgs/msg/PoseStamped',
                          serialization_format='cdr'),
            TopicMetadata(name='imu',  type='sensor_msgs/msg/Imu',
                          serialization_format='cdr')
        ):
            self.writer.create_topic(topic)

        self.get_logger().info('Recording ready – waiting for joystick command.')

    # ---------- main synced callback ---------------------------------------------
    def sensor_callback(self, odom_msg, scan_msg, pose_msg, imu_msg):
        # Gate on recent joystick activity
        if self.last_cmd_time is None:                    # never saw active /joy
            return
        age = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        if age > self.record_timeout:                     # command too old
            return

        # If we reach here, a fresh joystick command is active – write to bag
        timestamp = self.get_clock().now().nanoseconds
        try:
            self.writer.write('odom', serialize_message(odom_msg),  timestamp)
            self.writer.write('scan', serialize_message(scan_msg),  timestamp)
            self.writer.write('pose', serialize_message(pose_msg),  timestamp)
            self.writer.write('imu',  serialize_message(imu_msg),   timestamp)

            self.msg_counter += 1
            if self.msg_counter % 10 == 0:
                self.get_logger().info(f'Writing data… ({self.msg_counter})')
        except Exception as e:
            self.get_logger().error(f'Recording error: {e}')

# -------------------------- boilerplate -----------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.writer:
            node.writer.close()
        node.get_logger().info(f'Done. Total messages recorded: {node.msg_counter}')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
