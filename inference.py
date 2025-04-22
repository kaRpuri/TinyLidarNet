#!/usr/bin/env python3
import os
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import tensorflow as tf
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped


class AutonomousNode(Node):
    def __init__(self):
        super().__init__('autonomous_node')

        # --- Load TFLite model and inspect input shape ---
        model_path = os.path.join(
            os.path.dirname(__file__),
            'Models/RNN_Attn_Controller.tflite'
        )
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        inp_details = self.interpreter.get_input_details()[0]
        _, seq_len, num_ranges, num_ch = inp_details['shape']
        self.seq_len = int(seq_len)       # e.g. 5
        self.num_ranges = int(num_ranges) # e.g. 406
        self.num_ch = int(num_ch)         # should be 2

        # ring buffers for scans + timestamps
        self.buff = deque(maxlen=self.seq_len)
        self.buff_ts = deque(maxlen=self.seq_len + 1)

        self.input_index = inp_details['index']
        self.output_details = self.interpreter.get_output_details()

        # parameters
        self.is_joy = self.declare_parameter('is_joy', False).value
        self.hz = 40
        self.period = 1.0 / self.hz
        self.prev_button = 0

        # QoS for best‑effort LiDAR
        lidar_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Subscribers
        self.create_subscription(Joy, '/joy',
                                 self.button_callback, 10)
        self.create_subscription(LaserScan, '/scan',
                                 self.lidar_callback, lidar_qos)
        self.create_subscription(PoseStamped, '/pf/viz/inferred_pose',
                                 self.odom_callback, 10)

        # Publisher
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        # Spin timer
        self.timer = self.create_timer(self.period,
                                       self.control_loop)
        self.start_ts = time.time()
        self.total_distance = 0.0
        self.start_position = None

        self.get_logger().info(f'AutonomousNode ready: seq_len={self.seq_len}, '
                               f'ranges={self.num_ranges}, ch={self.num_ch}')

    def lidar_callback(self, msg: LaserScan):
        # Clean and subsample
        ranges = np.nan_to_num(msg.ranges,
                              nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.linspace(0, len(ranges)-1,
                          self.num_ranges, dtype=int)
        scan = ranges[idx].astype(np.float32)

        # push into buffers
        self.buff.append(scan)
        # use ROS timestamp if available: msg.header.stamp.sec + nanosec * 1e-9
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.buff_ts.append(t)

    def button_callback(self, msg: Joy):
        # toggle manual override on A button (index 0)
        curr = msg.buttons[0]
        if curr == 1 and curr != self.prev_button:
            new_val = not self.get_parameter('is_joy').value
            self.set_parameters([rclpy.parameter.Parameter(
                'is_joy',
                rclpy.Parameter.Type.BOOL,
                new_val
            )])
        self.prev_button = curr

    def odom_callback(self, msg: PoseStamped):
        pos = [msg.pose.position.x, msg.pose.position.y]
        if self.start_position is None:
            self.start_position = pos
            return
        dist = np.linalg.norm(np.array(pos) -
                              np.array(self.start_position))
        self.total_distance += dist
        self.start_position = pos

    def dnn_output(self):
        # wait until we have seq_len scans
        if len(self.buff) < self.seq_len:
            return 0.0, 0.0

        # 1) stack scans: (seq_len, num_ranges)
        scans = np.stack(self.buff, axis=0)

        # 2) Δt array: seq_len entries
        ts = np.array(self.buff_ts)
        dt = np.diff(ts).reshape(self.seq_len, 1)

        # 3) tile dt → (seq_len, num_ranges)
        dt_tiled = np.repeat(dt, self.num_ranges, axis=1)

        # 4) build channels → (seq_len, num_ranges, 2)
        seq = np.stack([scans, dt_tiled], axis=2)

        # 5) batch dim → (1, seq_len, num_ranges, 2)
        inp = seq[None, ...].astype(np.float32)

        # 6) inference
        self.interpreter.set_tensor(self.input_index, inp)
        t0 = time.time()
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(
            self.output_details[0]['index'])[0]
        inf_ms = (time.time() - t0)*1000
        self.get_logger().debug(f'Inference: {inf_ms:.1f} ms')
        return float(out[0]), float(out[1])

    def control_loop(self):
        manual = self.get_parameter('is_joy').value
        self.get_logger().info(
            f'Manual={"ON" if manual else "OFF"}  '
            f'Dist={self.total_distance:.2f}m'
        )

        if not manual:
            steer, speed = self.dnn_output()

            # un-normalize to real-world range
            speed = self.linear_map(speed,
                                    0, 1,   # trained 0→1
                                    -0.5, 7.0)
            steer = self.linear_map(steer,
                                    -1, 1,  # trained -1→1
                                    -0.34, 0.34)

            # publish AckermannDriveStamped
            msg = AckermannDriveStamped()
            msg.drive.speed = speed / 4.0
            msg.drive.steering_angle = steer
            self.drive_pub.publish(msg)

        # timer deadline check
        dur = time.time() - self.start_ts
        if dur > self.period:
            self.get_logger().warn(
                f'Deadline miss: {dur*1000:.1f} ms'
            )
        self.start_ts = time.time()

    @staticmethod
    def linear_map(x, x_min, x_max, y_min, y_max):
        return ((x - x_min) /
                (x_max - x_min) *
                (y_max - y_min) +
                y_min)


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
