# data_collection.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from rclpy.serialization import serialize_message
import message_filters
import numpy as np
from sensor_msgs.msg import Joy, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('data_collection_node')
        self.pressed = False
        self.writer = None
        
        # ROS2 QoS Configuration
        lidar_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Create subscribers
        self.drive_sub = message_filters.Subscriber(
            self, AckermannDriveStamped, '/vesc/low_level/ackermann_cmd_mux/input/teleop'
        )
        self.lid_sub = message_filters.Subscriber(
            self, LaserScan, '/scan_filtered', qos_profile=lidar_qos
        )
        
        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.drive_sub, self.lid_sub],
            queue_size=10,
            slop=0.01
        )
        self.ts.registerCallback(self.sync_callback)
        
        # Joystick subscriber
        self.create_subscription(Joy, '/vesc/joy', self.button_callback, 10)

    def init_bag_writer(self):
        storage_options = StorageOptions(
            uri='Dataset/out_bag',
            storage_id='sqlite3'
        )
        converter_options = ConverterOptions('', '')
        
        self.writer = SequentialWriter()
        self.writer.open(storage_options, converter_options)
        
        # Define topics metadata
        ackermann_topic = TopicMetadata(
            name='Ackermann',
            type='ackermann_msgs/msg/AckermannDriveStamped',
            serialization_format='cdr'
        )
        
        lidar_topic = TopicMetadata(
            name='Lidar',
            type='sensor_msgs/msg/LaserScan',
            serialization_format='cdr'
        )
        
        self.writer.create_topic(ackermann_topic)
        self.writer.create_topic(lidar_topic)

    def sync_callback(self, ack_msg, ldr_msg):
        if self.pressed and self.writer:
            try:
                self.writer.write(
                    'Ackermann',
                    serialize_message(ack_msg),
                    self.get_clock().now().nanoseconds
                )
                self.writer.write(
                    'Lidar',
                    serialize_message(ldr_msg),
                    self.get_clock().now().nanoseconds
                )
            except Exception as e:
                self.get_logger().error(f'Bag write error: {str(e)}')

    def button_callback(self, j):
        if j.buttons[1] == 1 and not self.pressed:
            self.pressed = True
            self.init_bag_writer()
            self.get_logger().info('Recording Started')
        elif j.buttons[1] == 0 and self.pressed:
            self.pressed = False
            self.writer.close()
            self.get_logger().info('Recording Stopped')

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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
