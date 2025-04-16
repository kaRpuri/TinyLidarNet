# inference.py
import time
import numpy as np
import rclpy
from rclpy.node import Node
import tensorflow as tf
from sensor_msgs.msg import LaserScan, Joy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

class AutonomousNode(Node):
    def __init__(self):
        super().__init__('autonomous_node')
        
        # Parameters and Variables
        self.subsample_lidar = 2
        self.model_name = './Models/TLN_M_noquantized.tflite'
        self.prev = 0
        self.curr = 0
        self.start_position = None
        self.total_distance = 0.0
        self.lidar_data = None
        self.is_joy = self.declare_parameter('is_joy', False).value
        self.hz = 40
        self.period = 1.0 / self.hz
        
        # ROS2 QoS Profiles
        lidar_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Publishers
        self.servo_pub = self.create_publisher(
            AckermannDriveStamped,
            '/vesc/low_level/ackermann_cmd_mux/input/teleop',
            10
        )
        
        # Subscribers
        self.create_subscription(
            Joy,
            '/vesc/joy',
            self.button_callback,
            10
        )
        
        self.create_subscription(
            LaserScan,
            '/scan_filtered',
            self.lidar_callback,
            lidar_qos
        )
        
        self.create_subscription(
            Float64,
            '/vesc/commands/motor/speed',
            self.rpm_callback,
            10
        )
        
        self.create_subscription(
            Odometry,
            '/vesc/odom',
            self.odom_callback,
            10
        )
        
        # Load TensorFlow Lite model
        self.interpreter = tf.lite.Interpreter(model_path=self.model_name)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()
        
        # Timer for control loop
        self.timer = self.create_timer(self.period, self.control_loop)
        self.start_ts = time.time()

    def lidar_callback(self, msg):
        ldata = msg.ranges[::self.subsample_lidar]
        ldata = np.expand_dims(ldata, axis=-1).astype(np.float32)
        self.lidar_data = np.expand_dims(ldata, axis=0)

    def button_callback(self, msg):
        self.curr = msg.buttons[0]
        if self.curr == 1 and self.curr != self.prev:
            new_value = not self.get_parameter('is_joy').value
            self.set_parameters([rclpy.parameter.Parameter('is_joy', rclpy.Parameter.Type.BOOL, new_value)])
        self.prev = self.curr

    def rpm_callback(self, msg):
        self.wheel_speed = msg.data

    def odom_callback(self, msg):
        current_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        if self.start_position is None:
            self.start_position = current_position
            return
        distance = np.linalg.norm(np.array(current_position) - np.array(self.start_position))
        self.total_distance += distance
        self.start_position = current_position

    def dnn_output(self):
        if self.lidar_data is None:
            return 0.0, 0.0
        
        self.interpreter.set_tensor(self.input_index, self.lidar_data)
        start_time = time.time()
        self.interpreter.invoke()
        inf_time = (time.time() - start_time) * 1000
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output[0, 0], output[0, 1]

    def control_loop(self):
        self.is_joy = self.get_parameter('is_joy').value
        self.get_logger().info(f'Manual Control: {"ON" if self.is_joy else "OFF"} | Distance: {self.total_distance:.2f}m')
        
        if not self.is_joy and self.lidar_data is not None:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            
            servo, speed = self.dnn_output()
            speed = self.linear_map(speed, 0, 1, -0.5, 7.0)
            
            msg.drive.speed = float(speed)
            msg.drive.steering_angle = float(servo)
            self.servo_pub.publish(msg)
            
            dur = time.time() - self.start_ts
            if dur > self.period:
                self.get_logger().warn(f"Deadline miss: {dur*1000:.1f}ms")
            else:
                self.get_logger().debug(f"Inference time: {dur*1000:.1f}ms")

    @staticmethod
    def linear_map(x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
