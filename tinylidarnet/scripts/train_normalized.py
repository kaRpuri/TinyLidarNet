#!/usr/bin/env python3


import os
import time
import warnings
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# ROS 2 bag imports
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# Keras model imports
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    BatchNormalization, Dropout,
    Bidirectional, LSTM, Dense, Attention,
    GlobalAveragePooling1D  # replaces tf.reduce_mean
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

#========================================================
# Utility functions
#========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min


def read_ros2_bag(bag_path):
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    lidar_data, servo_data, speed_data, timestamps = [], [], [], []
    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        t = t_ns * 1e-9
        if topic == 'scan':
            msg = deserialize_message(serialized_msg, LaserScan)
            cleaned = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar_data.append(cleaned[::2])
            timestamps.append(t)
        elif topic == 'odom':
            msg = deserialize_message(serialized_msg, Odometry)
            servo_data.append(msg.twist.twist.angular.z)
            speed_data.append(msg.twist.twist.linear.x)

    return (
        np.array(lidar_data),
        np.array(servo_data),
        np.array(speed_data),
        np.array(timestamps)
    )


def create_lidar_sequences(lidar_data, servo_data, speed_data, timestamps, sequence_length=5):
    X, y = [], []
    num_ranges = lidar_data.shape[1]
    for i in range(len(lidar_data) - sequence_length):
        frames = np.stack(lidar_data[i:i+sequence_length], axis=0)
        dt = np.diff(timestamps[i:i+sequence_length+1]).reshape(sequence_length, 1)
        dt_tiled = np.repeat(dt, num_ranges, axis=1)
        seq = np.concatenate([frames[...,None], dt_tiled[...,None]], axis=2)
        X.append(seq)
        y.append([servo_data[i+sequence_length], speed_data[i+sequence_length]])
    return np.array(X), np.array(y)

def build_spatiotemporal_model(seq_len, num_ranges):
    inp = Input(shape=(seq_len, num_ranges, 2), name='lidar_sequence')

    # Conv block 1
    x = TimeDistributed(Conv1D(24, 10, strides=4, activation='relu'))(inp)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)

    # Conv block 2
    x = TimeDistributed(Conv1D(36, 8, strides=4, activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)

    # Conv block 3
    x = TimeDistributed(Conv1D(48, 4, strides=2, activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)

    # Flatten and LSTM
    x = TimeDistributed(Flatten())(x)
    lstm_out = Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)

    # Attention
    q = Dense(64)(lstm_out)
    k = Dense(64)(lstm_out)
    v = Dense(64)(lstm_out)
    attn = Attention()([q, v, k])

    context = GlobalAveragePooling1D()(attn)
    context = Dropout(0.3)(context)

    out = Dense(2, activation='tanh', name='controls')(context)
    return Model(inp, out, name='RNN_Attention_Controller')

# Learning rate schedule
initial_lr = 5e-5

def lr_schedule(epoch):
    return initial_lr if epoch < 10 else initial_lr * 0.1

#========================================================
# Main training script
#========================================================
if __name__ == '__main__':
    print('GPU AVAILABLE:', bool(tf.config.list_physical_devices('GPU')))

    bag_paths = [
        '/home/nvidia/f1tenth_ws/src/TinyLidarNet/tinylidarnet/scripts/sim_Dataset/test_levine4/test_levine4_0.db3'
    ]
    seq_len, batch_size, epochs = 5, 64, 20

    # Load data
    all_lidar, all_servo, all_speed, all_ts = [], [], [], []
    for p in bag_paths:
        l, s, sp, ts = read_ros2_bag(p)
        print(f'Loaded {len(l)} scans from {p}')
        all_lidar.extend(l); all_servo.extend(s)
        all_speed.extend(sp); all_ts.extend(ts)

    all_lidar = np.array(all_lidar)
    all_servo = np.array(all_servo)
    all_speed = np.array(all_speed)
    all_ts    = np.array(all_ts)

    # Normalize speed to [0,1]
    vmin, vmax = all_speed.min(), all_speed.max()
    all_speed = linear_map(all_speed, vmin, vmax, 0, 1)

    # Build sequences
    X, y = create_lidar_sequences(all_lidar, all_servo, all_speed, all_ts, seq_len)
    n_samples, _, num_ranges, _ = X.shape
    print(f'Total sequences: {n_samples}, ranges: {num_ranges}')

    # Shuffle & split
    X, y = shuffle(X, y, random_state=42)
    split = int(0.85 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build & compile model
    model = build_spatiotemporal_model(seq_len, num_ranges)
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='huber')
    print(model.summary())

    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        LearningRateScheduler(lr_schedule)
    ]

    # Train
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    print(f'Training done in {int(time.time() - start)}s')

    # Plot loss curve
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig('Figures/loss_curve.png')
    plt.close()

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    os.makedirs('Models', exist_ok=True)
    with open('Models/RNN_Attn_Controller.tflite', 'wb') as f:
        f.write(tflite_model)
    print('TFLite model saved.')

    # Final evaluation
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Final test loss: {test_loss:.4f}')
