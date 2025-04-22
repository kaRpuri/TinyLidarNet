import rclpy
import numpy as np
import tensorflow as tf
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class TemporalLidarDriver(Node):
    def __init__(self):
        super().__init__('temporal_lidar_driver')
        
        # Model configuration
        self.sequence_length = 20  # Match your trained model
        self.subsample_factor = 2
        self.model_path = './Models/temporal_lidarnet.tflite'
        self.lidar_buffer = []
        
        # ROS2 setup
        lidar_qos = QoSProfile(
            depth=self.sequence_length+1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Publishers and Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, lidar_qos)
        
        # TensorFlow Lite initialization
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Control timer (40Hz)
        self.create_timer(0.025, self.control_loop)

    def lidar_callback(self, msg):
        """Process and buffer LiDAR scans"""
        # Clean and subsample data
        cleaned = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
        subsampled = cleaned[::self.subsample_factor]
        
        # Maintain sequence buffer
        self.lidar_buffer.append(subsampled)
        if len(self.lidar_buffer) > self.sequence_length:
            self.lidar_buffer.pop(0)

    def prepare_input(self):
        """Create model input from buffer"""
        if len(self.lidar_buffer) < self.sequence_length:
            return None
            
        # Normalize and format sequence
        sequence = np.array(self.lidar_buffer[-self.sequence_length:]) / 30.0
        return sequence.reshape(1, self.sequence_length, 540, 1).astype(np.float32)

    def control_loop(self):
        """Main control processing"""
        model_input = self.prepare_input()
        if model_input is None:
            self.get_logger().info("Initializing LiDAR buffer...", throttle_duration_sec=1)
            return
            
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], model_input)
        self.interpreter.invoke()
        outputs = self.interpreter.get_tensor(self.output_details[0]['index']).squeeze()
        
        # Publish controls
        cmd = Twist()
        cmd.angular.z = float(outputs[0])  # Steering [-1.0, 1.0]
        cmd.linear.x = float(outputs[1])   # Speed [0.0, 1.0]
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    driver = TemporalLidarDriver()
    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
