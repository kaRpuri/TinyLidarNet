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

# TensorRT + PyCUDA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver

from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(plan_path: str) -> trt.ICudaEngine:
    """Load a serialized TensorRT engine from disk."""
    with open(plan_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine: trt.ICudaEngine):
    """Allocate host and device buffers for all bindings."""
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for idx in range(engine.num_bindings):
        name = engine.get_binding_name(idx)
        is_input = engine.binding_is_input(idx)
        shape = engine.get_binding_shape(idx)
        dtype = trt.nptype(engine.get_binding_dtype(idx))
        # volume = batch * C * H * W for 4D, or just product(shape) if dynamic
        volume = int(np.prod(shape))
        # host (page-locked) and device buffers
        host_mem = cuda.pagelocked_empty(volume, dtype)
        dev_mem  = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(dev_mem))
        (inputs if is_input else outputs).append({
            'index': idx,
            'name': name,
            'host_mem': host_mem,
            'dev_mem': dev_mem,
            'shape': shape
        })
    return inputs, outputs, bindings, stream

class AutonomousNode(Node):
    def __init__(self):
        super().__init__('autonomous_node_trt')

        # ——— Load TensorRT engine ———
        plan_path = "/home/shyryn/sample/model_int8.plan"
        if not os.path.isfile(plan_path):
            self.get_logger().fatal(f"Engine not found: {plan_path}")
            raise FileNotFoundError(plan_path)

        self.engine  = load_engine(plan_path)
        self.context = self.engine.create_execution_context()
        (self.trt_inputs,
         self.trt_outputs,
         self.bindings,
         self.stream) = allocate_buffers(self.engine)

        # We assume single input, single output
        inp = self.trt_inputs[0]
        self.seq_len, self.num_ranges, self.num_ch = inp['shape']
        self.get_logger().info(
            f"TRT engine loaded. Input binding '{inp['name']}' shape={inp['shape']}, "
            f"output '{self.trt_outputs[0]['name']}' shape={self.trt_outputs[0]['shape']}"
        )

        # Thread-safe ring buffers for LiDAR & timestamps
        self.lock       = Lock()
        self.buff_scans = deque(maxlen=self.seq_len)
        self.buff_ts    = deque(maxlen=self.seq_len)

        # Manual override via joystick
        self.declare_parameter('is_joy', False)
        self.prev_button = 0

        # Distance tracking
        self.total_distance = 0.0
        self.start_position = None

        # ROS2 QoS & pubs/subs
        lidar_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.create_subscription(Joy, '/joy', self.button_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, lidar_qos)
        self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Control at 40 Hz
        self.timer = self.create_timer(1/40.0, self.control_loop)
        self.get_logger().info("AutonomousNode (TRT) ready.")

    def lidar_callback(self, msg: LaserScan):
        cleaned = np.nan_to_num(msg.ranges, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.linspace(0, len(cleaned)-1, self.num_ranges, dtype=int)
        scan = cleaned[idx].astype(np.float32)
        with self.lock:
            self.buff_scans.append(scan)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.buff_ts.append(t)

    def button_callback(self, msg: Joy):
        curr = msg.buttons[0]
        if curr == 1 and curr != self.prev_button:
            new = not self.get_parameter('is_joy').value
            self.set_parameters([Parameter('is_joy', Parameter.Type.BOOL, new)])
        self.prev_button = curr

    def odom_callback(self, msg: PoseStamped):
        pos = np.array([msg.pose.position.x, msg.pose.position.y])
        if self.start_position is None:
            self.start_position = pos
        else:
            self.total_distance += np.linalg.norm(pos - self.start_position)
            self.start_position = pos

    def dnn_output(self):
        # Wait for seq_len scans
        with self.lock:
            if len(self.buff_scans) < self.seq_len:
                return 0.0, 0.0
            scans = np.stack(self.buff_scans, axis=0)
            ts    = np.array(self.buff_ts, dtype=np.float32)

        # compute Δt, pad first with zero
        diffs = np.diff(ts)
        dt = np.zeros(self.seq_len, np.float32)
        dt[1:] = diffs.clip(min=0)

        # tile and stack → shape (seq_len, num_ranges, 2)
        dt_tiled = np.repeat(dt[:,None], self.num_ranges, axis=1)
        seq      = np.stack([scans, dt_tiled], axis=2)

        # copy into TRT input host buffer
        inp_buf = self.trt_inputs[0]
        flat    = seq.ravel()
        np.copyto(inp_buf['host_mem'], flat)

        # transfer to GPU, execute, transfer back
        cuda.memcpy_htod_async(inp_buf['dev_mem'], inp_buf['host_mem'], self.stream)
        start = time.time()
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.trt_outputs[0]['dev_mem'], self.trt_outputs[0]['host_mem'], self.stream)
        self.stream.synchronize()
        inf_ms = (time.time() - start)*1000.0

        out = self.trt_outputs[0]['host_mem']
        steer, speed = float(out[0]), float(out[1])
        self.get_logger().debug(f"TRT inf: {inf_ms:.2f} ms → steer={steer:.3f}, speed={speed:.3f}")
        return steer, speed

    @staticmethod
    def linear_map(x, xmin, xmax, ymin, ymax):
        return (x - xmin)/(xmax - xmin)*(ymax - ymin) + ymin

    def control_loop(self):
        joy = self.get_parameter('is_joy').value
        self.get_logger().info(f"Manual={'ON' if joy else 'OFF'} | Dist={self.total_distance:.2f}m")
        if not joy:
            steer, speed = self.dnn_output()
            # reverse normalization
            speed = self.linear_map(speed, 0.0, 1.0, -0.5, 7.0)
            steer = self.linear_map(steer, -1.0, 1.0, -0.34, 0.34)
            msg = AckermannDriveStamped()
            msg.drive.speed = speed/4.0
            msg.drive.steering_angle = steer
            self.drive_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
