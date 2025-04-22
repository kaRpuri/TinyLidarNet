# #!/usr/bin/env python3
# import os
# import time
# from collections import deque

# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.parameter import Parameter
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# import tensorflow as tf
# from sensor_msgs.msg import LaserScan, Joy
# from geometry_msgs.msg import PoseStamped
# from ackermann_msgs.msg import AckermannDriveStamped


# class AutonomousNode(Node):
#     def __init__(self):
#         super().__init__('autonomous_node')

#         # ——— Load the TFLite model ———
#         model_path = os.path.join(
#             os.path.dirname(__file__),
#             'Models/RNN_Attn_Controller.tflite'
#         )
#         self.interpreter = tf.lite.Interpreter(model_path=model_path)
#         self.interpreter.allocate_tensors()

#         # Inspect input shape: (batch, seq_len, num_ranges, num_ch)
#         inp = self.interpreter.get_input_details()[0]
#         _, seq_len, num_ranges, num_ch = inp['shape']
#         self.seq_len = int(seq_len)
#         self.num_ranges = int(num_ranges)
#         self.num_ch = int(num_ch)  # should be 2 after your training fix

#         self.input_index = inp['index']
#         self.output_details = self.interpreter.get_output_details()

#         # ——— Ring buffers for the last seq_len scans & timestamps ———
#         self.buff_scans = deque(maxlen=self.seq_len)       # holds np.array(scan)
#         self.buff_ts    = deque(maxlen=self.seq_len)       # one timestamp per scan

#         # Manual override parameter
#         self.declare_parameter('is_joy', False)
#         self.prev_button = 0

#         # Loop timing
#         self.hz = 40.0
#         self.period = 1.0 / self.hz
#         self.start_ts = time.time()
#         self.total_distance = 0.0
#         self.start_position = None

#         # QoS for LiDAR (best-effort)
#         lidar_qos = QoSProfile(
#             depth=10,
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             durability=QoSDurabilityPolicy.VOLATILE
#         )

#         # ——— ROS2 Subscriptions & Publishers ———
#         self.create_subscription(
#             Joy, '/joy', self.button_callback, 10
#         )
#         self.create_subscription(
#             LaserScan, '/scan', self.lidar_callback, lidar_qos
#         )
#         self.create_subscription(
#             PoseStamped, '/pf/viz/inferred_pose',
#             self.odom_callback, 10
#         )
#         self.drive_pub = self.create_publisher(
#             AckermannDriveStamped, '/drive', 10
#         )

#         # Timer for control loop
#         self.create_timer(self.period, self.control_loop)

#         self.get_logger().info(
#             f'Node ready: seq_len={self.seq_len}, '
#             f'ranges={self.num_ranges}, channels={self.num_ch}'
#         )

#     def lidar_callback(self, msg: LaserScan):
#         # Clean NaN/Inf and subsample to num_ranges points
#         cleaned = np.nan_to_num(msg.ranges, nan=0.0, posinf=0.0, neginf=0.0)
#         idx = np.linspace(0, len(cleaned)-1, self.num_ranges, dtype=int)
#         scan = cleaned[idx].astype(np.float32)

#         # Push scan + timestamp
#         self.buff_scans.append(scan)
#         t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
#         self.buff_ts.append(t)

#     def button_callback(self, msg: Joy):
#         curr = msg.buttons[0]  # A button
#         if curr == 1 and curr != self.prev_button:
#             new_val = not self.get_parameter('is_joy').value
#             self.set_parameters([Parameter('is_joy',
#                                            Parameter.Type.BOOL,
#                                            new_val)])
#         self.prev_button = curr

#     def odom_callback(self, msg: PoseStamped):
#         pos = np.array([msg.pose.position.x, msg.pose.position.y])
#         if self.start_position is None:
#             self.start_position = pos
#             return
#         self.total_distance += np.linalg.norm(pos - self.start_position)
#         self.start_position = pos

#     def dnn_output(self):
#         # Need exactly seq_len scans to run inference
#         if len(self.buff_scans) < self.seq_len:
#             return 0.0, 0.0

#         # 1) Stack scans: shape (seq_len, num_ranges)
#         scans = np.stack(self.buff_scans, axis=0)

#         # 2) Build Δt array
#         ts = np.array(self.buff_ts)  # length = seq_len
#         diffs = np.diff(ts)          # length = seq_len-1

#         # Pad to length seq_len by prepending a zero
#         dt_full = np.zeros(self.seq_len, dtype=np.float32)
#         dt_full[1:] = diffs

#         # 3) Tile Δt across ranges: (seq_len, num_ranges)
#         dt_tiled = np.repeat(dt_full[:, None], self.num_ranges, axis=1)

#         # 4) Build model input: (seq_len, num_ranges, 2)
#         seq = np.stack([scans, dt_tiled], axis=2)

#         # 5) Add batch dim: (1, seq_len, num_ranges, 2)
#         inp = seq[None, ...].astype(np.float32)

#         # 6) Run TFLite
#         self.interpreter.set_tensor(self.input_index, inp)
#         start = time.time()
#         self.interpreter.invoke()
#         out = self.interpreter.get_tensor(
#             self.output_details[0]['index']
#         )[0]
#         inf_ms = (time.time() - start) * 1000.0
#         self.get_logger().debug(f'Inference: {inf_ms:.1f} ms')

#         return float(out[0]), float(out[1])

#     @staticmethod
#     def linear_map(x, x_min, x_max, y_min, y_max):
#         return (x - x_min)/(x_max - x_min)*(y_max - y_min) + y_min

#     def control_loop(self):
#         joy = self.get_parameter('is_joy').value
#         self.get_logger().info(
#             f'Manual: {"ON" if joy else "OFF"} | '
#             f'Dist: {self.total_distance:.2f} m'
#         )

#         if not joy:
#             steer, speed = self.dnn_output()

#             # Reverse training normalization:
#             speed = self.linear_map(speed, 0.0, 1.0, -0.5, 7.0)
#             steer = self.linear_map(steer, -1.0, 1.0, -0.34, 0.34)
#             print("steeromg:", steer)

#             msg = AckermannDriveStamped()
#             msg.drive.speed = speed / 3.0
#             msg.drive.steering_angle = steer
#             self.drive_pub.publish(msg)

#         # Deadline check
#         dur = time.time() - self.start_ts
#         if dur > self.period:
#             self.get_logger().warn(
#                 f'Deadline miss: {dur*1000:.1f} ms'
#             )
#         self.start_ts = time.time()

# def main(args=None):
#     rclpy.init(args=args)
#     node = AutonomousNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()



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

class AutonomousNode(Node):
    def __init__(self):
        super().__init__('autonomous_node')
        
        # How many points the model wants (540)
        self.exp_len = 406 #self.interpreter.get_input_details()[0]['shape'][1]

        # Parameters and Variables
        self.subsample_lidar = 2
        self.model_name = './Models/TLN_noquantized.tflite'
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

            msg.drive.speed = float(speed/3) 
            msg.drive.steering_angle = float(steering)
            
            #self.cmd_pub.publish(msg)
            self.drive_pub.publish(msg)
            
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


