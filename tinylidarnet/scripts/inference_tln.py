# inference.py
import time
import numpy as np
import rclpy
from rclpy.node import Node
import tensorflow as tf
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import Twist  # Changed from AckermannDriveStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped        # NEW
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import matplotlib.pyplot as plt

class AutonomousNode(Node):
    def __init__(self):
        super().__init__('autonomous_node')
        
        # How many points the model wants (540)
        self.exp_len = 406 #self.interpreter.get_input_details()[0]['shape'][1]

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
        
        # ROS2 QoS Profiles (Match data collection)
        lidar_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Publishers (Updated to Twist)
        #self.cmd_pub = self.create_publisher(
        #    Twist,          # Changed from AckermannDriveStamped
        #    '/cmd_vel',     # Match data collection topic
        #    10
        #)

        self.drive_pub = self.create_publisher(                     # NEW
            AckermannDriveStamped,
            '/drive',
            10
        )
        
        # Subscribers (Updated topics)
        self.create_subscription(
            Joy,
            '/joy',         # Match data collection topic
            self.button_callback,
            10
        )
        
        self.create_subscription(
            LaserScan,
            '/scan',        # Changed from /scan_filtered
            self.lidar_callback,
            lidar_qos
        )
        
        '''
        self.create_subscription(
            Odometry,
            '/odom',  # Match data collection topic
            self.odom_callback,
            10
        )
        '''

        self.create_subscription(
            PoseStamped,
            "/pf/viz/inferred_pose",  # Match data collection topic
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

        self.recorded_ts = []
        self.recorded_speed = []
        self.recorded_steer = []


    def lidar_callback(self, msg):
        # Handle NaN/inf values like in training

        # clean NaN / Inf
        ranges = np.nan_to_num(msg.ranges, nan=0.0, posinf=0.0, neginf=0.0)

        # ----- resample to self.exp_len points -----
        idx = np.linspace(0, len(ranges) - 1, self.exp_len, dtype=np.int32)
        ldata = ranges[idx].astype(np.float32)                # (540,)

        # add channel and batch dims ⇒ (1, 540, 1)
        self.lidar_data = ldata[None, :, None]

        '''
        cleaned_ranges = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
        ldata = cleaned_ranges[::self.subsample_lidar]
        ldata = np.expand_dims(ldata, axis=-1).astype(np.float32)
        self.lidar_data = np.expand_dims(ldata, axis=0)
        '''

    def button_callback(self, msg):
        # Use Button 0 (A) for control toggle
        self.curr = msg.buttons[0]  # Changed from buttons[1]
        if self.curr == 1 and self.curr != self.prev:
            new_value = not self.get_parameter('is_joy').value
            self.set_parameters([rclpy.parameter.Parameter('is_joy', rclpy.Parameter.Type.BOOL, new_value)])
        self.prev = self.curr

    def odom_callback(self, msg):
        current_position = [msg.pose.position.x, msg.pose.position.y]
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
            msg = AckermannDriveStamped()  # Changed from AckermannDriveStamped
            
            #msg = Twist()

            # Get and map outputs (match training normalization)
            steering, speed = self.dnn_output()
            
            # Reverse mapping from training
            speed = self.linear_map(speed, 0, 1, -0.5, 7.0)  # Match train.py params
            steering = self.linear_map(steering, -1, 1, -0.34, 0.34)  # Typical steering range
            
            #msg.linear.x = float(speed)
            #msg.angular.z = float(steering)  # Using angular.z for steering

            msg.drive.speed = float(speed) 
            msg.drive.steering_angle = float(steering)
            
            #self.cmd_pub.publish(msg)
            self.drive_pub.publish(msg)
            
            dur = time.time() - self.start_ts
            if dur > self.period:
                self.get_logger().warn(f"Deadline miss: {dur*1000:.1f}ms")
            else:
                self.get_logger().debug(f"Inference time: {dur*1000:.1f}ms")

            curr_time = time.time() - self.start_ts
            self.recorded_ts.append(curr_time)
            self.recorded_speed.append(msg.drive.speed)
            self.recorded_steer.append(msg.drive.steering_angle)


    @staticmethod
    def linear_map(x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
    
def plot_results(t_unused, speed, steer, filename="Figures/tln_speed_steering_plot.png"):
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    timesteps = range(len(speed))  # use array index as x-axis

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(timesteps, speed, label='Speed [m/s]')
    plt.ylabel("Speed")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timesteps, steer, label='Steering Angle [rad]', color='orange')
    plt.xlabel("Timestep")
    plt.ylabel("Steering")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plot_results(
            node.recorded_ts,
            node.recorded_speed,
            node.recorded_steer,
            filename="Figures/tln_speed_steering_plot.png"
        )
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


