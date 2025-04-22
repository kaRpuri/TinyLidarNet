#!/usr/bin/env python3
import os
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import tensorflow as tf
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped


class AutonomousNode(Node):
    def __init__(self):
        super().__init__('autonomous_node')

        # ——— Load the TFLite model ———
        model_path = os.path.join(
            os.path.dirname(__file__),
            'Models/RNN_Attn_Controller.tflite'
        )
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Inspect input shape: (batch, seq_len, num_ranges, num_ch)
        inp = self.interpreter.get_input_details()[0]
        _, seq_len, num_ranges, num_ch = inp['shape']
        self.seq_len = int(seq_len)
        self.num_ranges = int(num_ranges)
        self.num_ch = int(num_ch)  # should be 2 after your training fix

        self.input_index = inp['index']
        self.output_details = self.interpreter.get_output_details()

        # ——— Ring buffers for the last seq_len scans & timestamps ———
        self.buff_scans = deque(maxlen=self.seq_len)       # holds np.array(scan)
        self.buff_ts    = deque(maxlen=self.seq_len)       # one timestamp per scan

        # Manual override parameter
        self.declare_parameter('is_joy', False)
        self.prev_button = 0

        # Loop timing
        self.hz = 40.0
        self.period = 1.0 / self.hz
        self.start_ts = time.time()
        self.total_distance = 0.0
        self.start_position = None

        # QoS for LiDAR (best-effort)
        lidar_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # ——— ROS2 Subscriptions & Publishers ———
        self.create_subscription(
            Joy, '/joy', self.button_callback, 10
        )
        self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, lidar_qos
        )
        self.create_subscription(
            PoseStamped, '/pf/viz/inferred_pose',
            self.odom_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        # Timer for control loop
        self.create_timer(self.period, self.control_loop)

        self.get_logger().info(
            f'Node ready: seq_len={self.seq_len}, '
            f'ranges={self.num_ranges}, channels={self.num_ch}'
        )

    def lidar_callback(self, msg: LaserScan):
        # Clean NaN/Inf and subsample to num_ranges points
        cleaned = np.nan_to_num(msg.ranges, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.linspace(0, len(cleaned)-1, self.num_ranges, dtype=int)
        scan = cleaned[idx].astype(np.float32)

        # Push scan + timestamp
        self.buff_scans.append(scan)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.buff_ts.append(t)

    def button_callback(self, msg: Joy):
        curr = msg.buttons[0]  # A button
        if curr == 1 and curr != self.prev_button:
            new_val = not self.get_parameter('is_joy').value
            self.set_parameters([Parameter('is_joy',
                                           Parameter.Type.BOOL,
                                           new_val)])
        self.prev_button = curr

    def odom_callback(self, msg: PoseStamped):
        pos = np.array([msg.pose.position.x, msg.pose.position.y])
        if self.start_position is None:
            self.start_position = pos
            return
        self.total_distance += np.linalg.norm(pos - self.start_position)
        self.start_position = pos

    def dnn_output(self):
        # Need exactly seq_len scans to run inference
        if len(self.buff_scans) < self.seq_len:
            return 0.0, 0.0

        # 1) Stack scans: shape (seq_len, num_ranges)
        scans = np.stack(self.buff_scans, axis=0)

        # 2) Build Δt array
        ts = np.array(self.buff_ts)  # length = seq_len
        diffs = np.diff(ts)          # length = seq_len-1

        # Pad to length seq_len by prepending a zero
        dt_full = np.zeros(self.seq_len, dtype=np.float32)
        dt_full[1:] = diffs

        # 3) Tile Δt across ranges: (seq_len, num_ranges)
        dt_tiled = np.repeat(dt_full[:, None], self.num_ranges, axis=1)

        # 4) Build model input: (seq_len, num_ranges, 2)
        seq = np.stack([scans, dt_tiled], axis=2)

        # 5) Add batch dim: (1, seq_len, num_ranges, 2)
        inp = seq[None, ...].astype(np.float32)

        # 6) Run TFLite
        self.interpreter.set_tensor(self.input_index, inp)
        start = time.time()
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0]
        inf_ms = (time.time() - start) * 1000.0
        self.get_logger().debug(f'Inference: {inf_ms:.1f} ms')

        return float(out[0]), float(out[1])

    @staticmethod
    def linear_map(x, x_min, x_max, y_min, y_max):
        return (x - x_min)/(x_max - x_min)*(y_max - y_min) + y_min

    def control_loop(self):
        joy = self.get_parameter('is_joy').value
        self.get_logger().info(
            f'Manual: {"ON" if joy else "OFF"} | '
            f'Dist: {self.total_distance:.2f} m'
        )

        if not joy:
            steer, speed = self.dnn_output()

            # Reverse training normalization:
            speed = self.linear_map(speed, 0.0, 1.0, -0.5, 7.0)
            steer = self.linear_map(steer, -1.0, 1.0, -0.34, 0.34)

            msg = AckermannDriveStamped()
            msg.drive.speed = speed / 4.0
            msg.drive.steering_angle = steer
            self.drive_pub.publish(msg)

        # Deadline check
        dur = time.time() - self.start_ts
        if dur > self.period:
            self.get_logger().warn(
                f'Deadline miss: {dur*1000:.1f} ms'
            )
        self.start_ts = time.time()

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