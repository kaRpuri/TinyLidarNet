# train.py
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
from ackermann_msgs.msg import AckermannDriveStamped

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
# ROS2 Data Loading
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
        if topic == 'Lidar':
            msg = deserialize_message(data, LaserScan)
            lidar_data.append(msg.ranges[::down_sample_param])
        elif topic == 'Ackermann':
            msg = deserialize_message(data, AckermannDriveStamped)
            servo_data.append(msg.drive.steering_angle)
            speed_data.append(msg.drive.speed)
    
    return np.array(lidar_data), np.array(servo_data), np.array(speed_data)

#========================================================
# Main Training Code
#========================================================
if __name__ == "__main__":
    # Configuration
    model_name = 'TLN'
    dataset_path = ['./Dataset/out_bag', './Dataset/f2_bag', './Dataset/f4_bag']
    down_sample_param = 2
    lr = 5e-5
    batch_size = 64
    num_epochs = 20
    hz = 40

    # Data loading
    lidar, servo, speed = [], [], []
    test_lidar, test_servo, test_speed = [], [], []
    max_speed = 0
    min_speed = 0

    for pth in dataset_path:
        l_data, s_data, sp_data = read_ros2_bag(pth)
        lidar_data = l_data
        servo_data = s_data
        speed_data = sp_data

        lidar_data = np.array(lidar_data) 
        servo_data = np.array(servo_data)
        speed_data = np.array(speed_data)

        shuffled_data = shuffle(np.concatenate((servo_data[:, np.newaxis], speed_data[:, np.newaxis]), axis=1), random_state=62)
        shuffled_lidar_data = shuffle(lidar_data, random_state=62)

        train_ratio = 0.85
        train_samples = int(train_ratio * len(shuffled_lidar_data))
        x_train_bag, x_test_bag = shuffled_lidar_data[:train_samples], shuffled_lidar_data[train_samples:]
        y_train_bag = shuffled_data[:train_samples]
        y_test_bag = shuffled_data[train_samples:]

        lidar.extend(x_train_bag)
        servo.extend(y_train_bag[:, 0])
        speed.extend(y_train_bag[:, 1])
        test_lidar.extend(x_test_bag)
        test_servo.extend(y_test_bag[:, 0])
        test_speed.extend(y_test_bag[:, 1])

        print(f'\nData in {pth}:')
        print(f'Train Data --- Lidar: {len(lidar)}, Servo: {len(servo)}, Speed: {len(speed)}')
        print(f'Test Data --- Lidar: {len(test_lidar)}, Servo: {len(test_servo)}, Speed: {len(test_speed)}')

    total_number_samples = len(lidar)
    lidar = np.asarray(lidar)
    servo = np.asarray(servo)
    speed = np.asarray(speed)
    speed = linear_map(speed, min_speed, max_speed, 0, 1)
    test_lidar = np.asarray(test_lidar)
    test_servo = np.asarray(test_servo)
    test_speed = np.asarray(test_speed)
    test_speed = linear_map(test_speed, min_speed, max_speed, 0, 1)

    print(f'\nMin_speed: {min_speed}')
    print(f'Max_speed: {max_speed}')
    print(f'Loaded {len(lidar)} Training samples ({len(lidar)/total_number_samples*100:.2f}%)')
    print(f'Loaded {len(test_lidar)} Testing samples ({len(test_lidar)/total_number_samples*100:.2f}%)\n')

    # Model architecture
    num_lidar_range_values = len(lidar[0])
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
        np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1),
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(test_lidar, np.concatenate((test_servo[:, np.newaxis], test_speed[:, np.newaxis]), axis=1))
    )
    print(f'\nTraining completed in {int(time.time() - start_time)} seconds')

    # Plot training history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./Figures/loss_curve.png')
    plt.close()

    # Model conversion to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save models
    with open('./Models/TLN_noquantized.tflite', 'wb') as f:
        f.write(tflite_model)
    print('TFLite models saved successfully')

    # Evaluation
    test_loss = model.evaluate(test_lidar, np.concatenate((test_servo[:, np.newaxis], test_speed[:, np.newaxis]), axis=1))
    print(f'\nFinal test loss: {test_loss:.4f}')
