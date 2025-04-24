#!/usr/bin/env python3
import os
import time
from collections import deque
from threading import Lock

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

        self.buffer_lock = Lock()

        # === Load the TFLite model ===
        try:
            model_path = os.path.join(
                os.path.dirname(__file__),
                'Models/TLN_2DCNN.tflite'
            )

            if not os.path.exists(model_path):
                self.get_logger().error(f'Model file not found: {model_path}')
                raise FileNotFoundError(f'Model file not found: {model_path}')

            self.get_logger().info(f'Loading model from: {model_path}')
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            inp = self.interpreter.get_input_details()[0]
            inp_shape = inp['shape']
            self.input_index = inp['index']
            self.output_details = self.interpreter.get_output_details()

            # Detect 2D CNN input format
            if len(inp_shape) == 4:
                _, height, width, channels = inp_shape
                self.seq_len = height
                self.num_ranges = width
                self.num_ch = channels
            else:
                raise ValueError(f"Unsupported input shape: {inp_shape}")

            self.get_logger().info(f'Model expects input shape: {inp_shape}, dtype: {inp["dtype"]}')

        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

        # Buffers
        self.latest_scan = None
        self.timestamp = None

        # Parameters
        self.declare_parameter('is_joy', False)
        self.prev_button = 0

        self.hz = 40.0
        self.period = 1.0 / self.hz
        self.start_ts = time.time()
        self.total_distance = 0.0
        self.start_position = None

        lidar_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.create_subscription(Joy, '/joy', self.button_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, lidar_qos)
        self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_timer(self.period, self.control_loop)

        self.get_logger().info(f'Node ready: height={self.seq_len}, width={self.num_ranges}, channels={self.num_ch}')

    def lidar_callback(self, msg: LaserScan):
        try:
            cleaned = np.nan_to_num(msg.ranges, nan=0.0, posinf=0.0, neginf=0.0)
            idx = np.linspace(0, len(cleaned) - 1, self.num_ranges, dtype=int)
            scan = cleaned[idx].astype(np.float32)

            with self.buffer_lock:
                self.latest_scan = scan
                self.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        except Exception as e:
            self.get_logger().error(f'Error in lidar_callback: {e}')

    def button_callback(self, msg: Joy):
        try:
            curr = msg.buttons[0]  # A button
            if curr == 1 and curr != self.prev_button:
                new_val = not self.get_parameter('is_joy').value
                self.set_parameters([Parameter('is_joy', Parameter.Type.BOOL, new_val)])
            self.prev_button = curr
        except Exception as e:
            self.get_logger().error(f'Error in button_callback: {e}')

    def odom_callback(self, msg: PoseStamped):
        try:
            pos = np.array([msg.pose.position.x, msg.pose.position.y])
            if self.start_position is None:
                self.start_position = pos
                return
            self.total_distance += np.linalg.norm(pos - self.start_position)
            self.start_position = pos
        except Exception as e:
            self.get_logger().error(f'Error in odom_callback: {e}')

    def dnn_output(self):
        try:
            with self.buffer_lock:
                if self.latest_scan is None:
                    return 0.0, 0.0
                scan = self.latest_scan.copy()

            if np.isnan(scan).any() or np.isinf(scan).any():
                scan = np.nan_to_num(scan, nan=0.0, posinf=0.0, neginf=0.0)

            # Reshape for 2D CNN input: (1, 1, num_ranges, 1)
            inp = scan[None, None, :, None].astype(np.float32)

            self.interpreter.set_tensor(self.input_index, inp)
            start = time.time()
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            inf_ms = (time.time() - start) * 1000.0
            self.get_logger().debug(f'Inference: {inf_ms:.1f} ms, Output: {out}, shape: {out.shape}')

            return float(out[0]), float(out[1])
        except Exception as e:
            self.get_logger().error(f'Error in dnn_output: {e}')
            return 0.0, 0.0

    @staticmethod
    def linear_map(x, x_min, x_max, y_min, y_max):
        if x_max == x_min:
            return (y_max + y_min) / 2
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def control_loop(self):
        try:
            joy = self.get_parameter('is_joy').value
            self.get_logger().info(f'Manual: {"ON" if joy else "OFF"} | Dist: {self.total_distance:.2f} m')

            if not joy:
                steer, speed = self.dnn_output()

                speed = self.linear_map(speed, 0.0, 1.0, -0.5, 7.0)
                steer = self.linear_map(steer, -1.0, 1.0, -0.34, 0.34)

                msg = AckermannDriveStamped()
                msg.drive.speed = speed /2.0
                msg.drive.steering_angle = steer
                self.drive_pub.publish(msg)

            dur = time.time() - self.start_ts
            if dur > self.period:
                self.get_logger().warn(f'Deadline miss: {dur*1000:.1f} ms')
            self.start_ts = time.time()

        except Exception as e:
            self.get_logger().error(f'Error in control_loop: {e}')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = AutonomousNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Fatal error: {e}")
    except KeyboardInterrupt:
        print("Node stopped by keyboard interrupt")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
