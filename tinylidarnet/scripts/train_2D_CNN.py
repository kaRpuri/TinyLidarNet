import os
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from vesc_msgs.msg import VescImuStamped

def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def read_ros2_bag(bag_path):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    reader = SequentialReader()
    reader.open(storage_options, ConverterOptions('', ''))

    lidar_data, servo_data, speed_data = [], [], []
    imu_data, pose_data = [], []

    while reader.has_next():
        topic, data, _ = reader.read_next()

        if topic == 'scan':
            msg = deserialize_message(data, LaserScan)
            cleaned_ranges = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar_data.append(cleaned_ranges[::2])  # downsample 2x

        elif topic == 'odom':
            msg = deserialize_message(data, Odometry)
            servo_data.append(msg.twist.twist.angular.z)
            speed_data.append(msg.twist.twist.linear.x)

        elif topic == 'imu':
            msg = deserialize_message(data, VescImuStamped)
            imu_data.append([
                msg.imu.linear_acceleration.x,
                msg.imu.linear_acceleration.y,
                msg.imu.linear_acceleration.z,
                msg.imu.angular_velocity.x,
                msg.imu.angular_velocity.y,
                msg.imu.angular_velocity.z
            ])

        elif topic == 'pose':
            msg = deserialize_message(data, PoseStamped)
            pose_data.append([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])

    return (
        np.array(lidar_data),
        np.array(servo_data),
        np.array(speed_data),
        np.array(imu_data),
        np.array(pose_data)
    )

if __name__ == "__main__":
    # Config
    model_name = 'TLN_2DCNN'
    dataset_path = ['/home/nvidia/f1tenth_ws/src/TinyLidarNet/tinylidarnet/scripts/car_Dataset/controller_slow_5min/controller_slow_5min_0.db3']
    lr = 5e-5
    batch_size = 64
    num_epochs = 20
    epsilon = 1e-8

    lidar, servo, speed, imu, pose = [], [], [], [], []
    test_lidar, test_servo, test_speed, test_imu, test_pose = [], [], [], [], []

    for path in dataset_path:
        l, s, spd, imu_d, pose_d = read_ros2_bag(path)

        indices = np.arange(len(l))
        np.random.shuffle(indices)
        split = int(0.85 * len(indices))

        lidar.extend(l[indices[:split]])
        servo.extend(s[indices[:split]])
        speed.extend(spd[indices[:split]])
        imu.extend(imu_d[indices[:split]])
        pose.extend(pose_d[indices[:split]])

        test_lidar.extend(l[indices[split:]])
        test_servo.extend(s[indices[split:]])
        test_speed.extend(spd[indices[split:]])
        test_imu.extend(imu_d[indices[split:]])
        test_pose.extend(pose_d[indices[split:]])

    # Reshape to (batch, height, width, channels) for 2D CNN
    lidar = np.expand_dims(np.expand_dims(np.array(lidar), 1), -1)
    test_lidar = np.expand_dims(np.expand_dims(np.array(test_lidar), 1), -1)

    servo = np.array(servo)
    test_servo = np.array(test_servo)
    speed = np.array(speed)
    test_speed = np.array(test_speed)

    min_speed, max_speed = speed.min(), speed.max()
    if max_speed - min_speed < epsilon:
        speed = np.zeros_like(speed)
        test_speed = np.zeros_like(test_speed)
    else:
        speed = linear_map(speed, min_speed, max_speed, 0, 1)
        test_speed = linear_map(test_speed, min_speed, max_speed, 0, 1)

    imu = np.array(imu)
    pose = np.array(pose)
    test_imu = np.array(test_imu)
    test_pose = np.array(test_pose)

    imu_mean, imu_std = imu.mean(axis=0), imu.std(axis=0)
    pose_mean, pose_std = pose.mean(axis=0), pose.std(axis=0)

    imu = (imu - imu_mean) / (imu_std + epsilon)
    test_imu = (test_imu - imu_mean) / (imu_std + epsilon)
    pose = (pose - pose_mean) / (pose_std + epsilon)
    test_pose = (test_pose - pose_mean) / (pose_std + epsilon)

    aux_input = np.concatenate([imu, pose], axis=1)
    test_aux_input = np.concatenate([test_imu, test_pose], axis=1)

    print(f"Train lidar: {lidar.shape}, aux: {aux_input.shape}")
    print(f"Train targets: speed {speed.shape}, servo {servo.shape}")

    # Model definition
    lidar_input = tf.keras.Input(shape=(lidar.shape[1], lidar.shape[2], lidar.shape[3]), name='lidar')
    x1 = tf.keras.layers.Conv2D(24, (1, 10), strides=(1, 4), activation='relu')(lidar_input)
    x1 = tf.keras.layers.Conv2D(36, (1, 8), strides=(1, 4), activation='relu')(x1)
    x1 = tf.keras.layers.Conv2D(48, (1, 4), strides=(1, 2), activation='relu')(x1)
    x1 = tf.keras.layers.Conv2D(64, (1, 3), activation='relu')(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    aux_input_layer = tf.keras.Input(shape=(aux_input.shape[1],), name='aux')
    x2 = tf.keras.layers.Dense(64, activation='relu')(aux_input_layer)

    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    output = tf.keras.layers.Dense(2, activation='tanh')(x)

    model = tf.keras.Model(inputs=[lidar_input, aux_input_layer], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='huber')
    model.summary()

    history = model.fit(
        {'lidar': lidar, 'aux': aux_input},
        np.stack([servo, speed], axis=1),
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(
            {'lidar': test_lidar, 'aux': test_aux_input},
            np.stack([test_servo, test_speed], axis=1)
        )
    )

    os.makedirs('./Figures', exist_ok=True)
    os.makedirs('./Models', exist_ok=True)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.savefig('./Figures/loss_curve_2dcnn.png')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    with open('./Models/TLN_2DCNN.tflite', 'wb') as f:
        f.write(tflite_model)

    print("✅ Training complete. TFLite model saved.")

    loss = model.evaluate({'lidar': test_lidar, 'aux': test_aux_input},
                          np.stack([test_servo, test_speed], axis=1))
    print(f"Final Test Loss: {loss:.4f}")
