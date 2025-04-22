import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import serialize_message
import message_filters
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('sim_data_collector')
        self.msg_counter = 0  # Add message counter
        
        # Initialize bag writer immediately
        self.writer = None
        self.init_bag_writer()
        
        # Configure QoS for sensor data
        sensor_qos = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Initialize synchronized subscribers
        #self.drive_sub = message_filters.Subscriber(self, AckermannDriveStamped, "/drive")
        #self.drive_sub = message_filters.Subscriber(self, Twist, '/cmd_vel')
        # self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom')
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/pf/pose/odom')
        self.lidar_sub = message_filters.Subscriber(self, LaserScan, '/scan', qos_profile=sensor_qos)
        
        self.imu_sub = message_filters.Subscriber(self, Imu, '/sensors/imu/raw')
        self.position_sub = message_filters.Subscriber(self, PoseStamped, '/pf/viz/inferred_pose')

        # Setup time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            #[self.drive_sub, self.odom_sub, self.lidar_sub],
            [self.odom_sub, self.lidar_sub, self.imu_sub, self.position_sub],
            #[self.odom_sub, self.lidar_sub],
            #[self.lidar_sub],
            #[self.drive_sub],
            #[self.odom_sub],
            queue_size=20,
            slop=0.05,
            allow_headerless=True
        )
        self.ts.registerCallback(self.sensor_callback)
        
        self.get_logger().info('Data collector node initialized!')

    def init_bag_writer(self):
        storage_options = StorageOptions(
            uri='sim_Dataset/test_levine2',
            storage_id='sqlite3'
        )
        converter_options = ConverterOptions('', '')
        
        self.writer = SequentialWriter()
        self.writer.open(storage_options, converter_options)
        
        # Register topics
        topics = [
            #TopicMetadata(name='cmd_vel', type='geometry_msgs/msg/Twist', serialization_format='cdr'),
            TopicMetadata(name='odom', type='nav_msgs/msg/Odometry', serialization_format='cdr'),
            TopicMetadata(name='scan', type='sensor_msgs/msg/LaserScan', serialization_format='cdr'),
            TopicMetadata(name='pose', type='geometry_msgs/msg/PoseStamped', serialization_format='cdr'),
            TopicMetadata(name='imu', type='sensor_msgs/msg/Imu', serialization_format='cdr')
            #TopicMetadata(name='drive', type='ackermann_msgs/msg/AckermannDriveStamped', serialization_format='cdr')
            
        ]
        
        for topic in topics:
            self.writer.create_topic(topic)
        self.get_logger().info('Recording STARTED! Saving data to sim_Dataset/testrun')

    #def sensor_callback(self, cmd_msg, odom_msg, scan_msg):
    def sensor_callback(self, odom_msg, scan_msg, pose_msg, imu_msg):
    #def sensor_callback(self, scan_msg):
    #def sensor_callback(self, cmd_msg):
        try:
            timestamp = self.get_clock().now().nanoseconds
            #self.writer.write('drive', serialize_message(cmd_msg), timestamp)
            #self.writer.write('cmd_vel', serialize_message(cmd_msg), timestamp)
            self.writer.write('odom', serialize_message(odom_msg), timestamp)
            self.writer.write('scan', serialize_message(scan_msg), timestamp)
            self.writer.write('pose', serialize_message(pose_msg), timestamp)
            self.writer.write('imu', serialize_message(imu_msg), timestamp)
            
            # Update counter and print status every 10 messages
            self.msg_counter += 1
            if self.msg_counter % 10 == 0:
                self.get_logger().info(f'Writing data... ({self.msg_counter} messages recorded)')
                
        except Exception as e:
            self.get_logger().error(f'Recording error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    collector = DataCollectionNode()
    
    try:
        collector.get_logger().info('Starting data collection...')
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info('Keyboard interrupt received')
    finally:
        if collector.writer:
            collector.writer.close()
        collector.get_logger().info(f'Recording COMPLETE! Total messages: {collector.msg_counter}')
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
