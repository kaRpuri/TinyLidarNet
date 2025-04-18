import os
import time
import warnings
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist  # Changed from AckermannDriveStamped

# Check GPU availability
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
print('GPU AVAILABLE:', gpu_available)

#========================================================
# Functions
#========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return np.mean(loss)

#========================================================
# ROS2 Data Loading (Updated for actual collected data)
#========================================================
def read_ros2_bag(bag_path):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    reader = SequentialReader()
    reader.open(storage_options, ConverterOptions('', ''))
    
    lidar_data = []
    servo_data = []
    speed_data = []
    
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == 'scan':  # Changed from 'Lidar'
            msg = deserialize_message(data, LaserScan)
            # Handle NaN/inf values in LiDAR data
            cleaned_ranges = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar_data.append(cleaned_ranges[::2])
        elif topic == 'cmd_vel':  # Changed from 'Ackermann'
            msg = deserialize_message(data, Twist)
            servo_data.append(msg.angular.z)  # Steering from angular.z
            speed_data.append(msg.linear.x)   # Speed from linear.x
    
    return np.array(lidar_data), np.array(servo_data), np.array(speed_data)

#========================================================
# Main Training Code (Structure unchanged)
#========================================================
if __name__ == "__main__":
    # Configuration
    model_name = 'TLN'
    dataset_path = ['./sim_Dataset/testrun/testrun_0.db3', './sim_Dataset/testrun1/testrun1_0.db3', './sim_Dataset/testrun3/testrun3_0.db3']
    down_sample_param = 2
    lr = 5e-5
    batch_size = 64
    num_epochs = 20
    hz = 40

    # Data loading
    lidar, servo, speed = [], [], []
    test_lidar, test_servo, test_speed = [], [], []

    for pth in dataset_path:
        l_data, s_data, sp_data = read_ros2_bag(pth)
        
        # Calculate actual speed range
        max_speed = np.max(sp_data)
        min_speed = np.min(sp_data)
        
        # Shuffle while maintaining alignment
        indices = np.arange(len(l_data))
        np.random.shuffle(indices)
        
        train_ratio = 0.85
        split_idx = int(len(indices) * train_ratio)
        
        # Training data
        lidar.extend(l_data[indices[:split_idx]])
        servo.extend(s_data[indices[:split_idx]])
        speed.extend(sp_data[indices[:split_idx]])
        
        # Test data
        test_lidar.extend(l_data[indices[split_idx:]])
        test_servo.extend(s_data[indices[split_idx:]])
        test_speed.extend(sp_data[indices[split_idx:]])

        print(f'\nData in {pth}:')
        print(f'Train Samples: {len(indices[:split_idx])}')
        print(f'Test Samples: {len(indices[split_idx:])}')

    # Convert to numpy arrays and add channel dimension
    lidar = np.expand_dims(np.asarray(lidar), -1)
    test_lidar = np.expand_dims(np.asarray(test_lidar), -1)
    
    # Normalize speed using actual values
    speed = linear_map(np.asarray(speed), min_speed, max_speed, 0, 1)
    test_speed = linear_map(np.asarray(test_speed), min_speed, max_speed, 0, 1)
    
    # Servo already in [-1,1] range for Twist.angular.z
    servo = np.asarray(servo)
    test_servo = np.asarray(test_servo)

    print(f'\nSpeed Range: {min_speed:.2f} to {max_speed:.2f}')
    print(f'Final Dataset Shapes:')
    print(f'Train Lidar: {lidar.shape}, Servo: {servo.shape}, Speed: {speed.shape}')
    print(f'Test Lidar: {test_lidar.shape}, Servo: {test_servo.shape}, Speed: {test_speed.shape}\n')

    # Model architecture (unchanged)
    num_lidar_range_values = lidar.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(24, 10, strides=4, activation='relu', input_shape=(num_lidar_range_values, 1)),
        tf.keras.layers.Conv1D(36, 8, strides=4, activation='relu'),
        tf.keras.layers.Conv1D(48, 4, strides=2, activation='relu'),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(2, activation='tanh')
    ])

    # Model compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='huber')
    print(model.summary())

    # Training
    start_time = time.time()
    history = model.fit(
        lidar,
        np.stack([servo, speed], axis=1),
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(test_lidar, np.stack([test_servo, test_speed], axis=1))
    )
    print(f'\nTraining completed in {int(time.time() - start_time)} seconds')

    # Plotting and saving (unchanged)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./Figures/loss_curve.png')
    plt.close()

    # Model conversion and saving (unchanged)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('./Models/TLN_noquantized.tflite', 'wb') as f:
        f.write(tflite_model)
    print('TFLite models saved successfully')

    # Evaluation
    test_loss = model.evaluate(test_lidar, np.stack([test_servo, test_speed], axis=1))
    print(f'\nFinal test loss: {test_loss:.4f}')
