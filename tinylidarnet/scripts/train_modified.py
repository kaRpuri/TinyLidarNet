#!/usr/bin/env python3
import os
import time
import numpy as np
import rosbag
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min


def huber_loss_np(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return np.mean(loss)


def create_lidar_sequences(lidar_data, servo_data, speed_data, timestamps, sequence_length=5):
    """
    Build sliding-window sequences of LiDAR frames with time-delta features
    and corresponding steering/speed targets.
    """
    X, y = [], []
    for i in range(len(lidar_data) - sequence_length):
        frames = np.stack(lidar_data[i:i + sequence_length], axis=0)
        dt = np.diff(timestamps[i:i + sequence_length + 1]).reshape(sequence_length, 1)
        seq = np.concatenate([frames, dt], axis=1)
        X.append(seq)
        y.append([servo_data[i + sequence_length], speed_data[i + sequence_length]])
    return np.array(X), np.array(y)

from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    Bidirectional, LSTM, Dense, Attention
)
from tensorflow.keras.models import Model


def build_spatiotemporal_model(seq_len, num_ranges):
    inputs = Input(shape=(seq_len, num_ranges, 1), name='lidar_sequence')
    x = TimeDistributed(Conv1D(24, 10, strides=4, activation='relu'))(inputs)
    x = TimeDistributed(Conv1D(36, 8, strides=4, activation='relu'))(x)
    x = TimeDistributed(Conv1D(48, 4, strides=2, activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    q = Dense(64)(lstm_out)
    k = Dense(64)(lstm_out)
    v = Dense(64)(lstm_out)
    attn = Attention()([q, v, k])
    context = tf.reduce_mean(attn, axis=1)
    outputs = Dense(2, activation='tanh', name='controls')(context)
    return Model(inputs, outputs, name='RNN_Attention_Controller')


def evaluate_tflite(model_path, X_data, y_data, seq_len, num_ranges):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']
    times, preds = [], []
    for sample in X_data:
        inp = sample.astype(np.float32).reshape(1, seq_len, num_ranges, 1)
        t0 = time.time()
        interp.set_tensor(inp_idx, inp)
        interp.invoke()
        out = interp.get_tensor(out_idx)[0]
        times.append((time.time() - t0) * 1e6)
        preds.append(out)
    preds = np.array(preds)
    print(f"TFLite Model: {os.path.basename(model_path)}")
    print(" - Huber Loss:", huber_loss_np(y_data, preds))
    print(" - Avg Inference Time (µs):", np.mean(times))
    return preds, times

if __name__ == '__main__':
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print('GPU AVAILABLE:', bool(gpus))

    bag_paths = ['./Dataset/out.bag', './Dataset/f2.bag', './Dataset/f4.bag']
    seq_len = 5
    batch_size = 64
    lr = 5e-5

    raw_lidar, raw_servo, raw_speed, timestamps = [], [], [], []
    for pth in bag_paths:
        if not os.path.exists(pth): raise FileNotFoundError(pth)
        bag = rosbag.Bag(pth)
        for topic, msg, t in bag.read_messages():
            if topic == 'Lidar':
                raw_lidar.append(np.array(msg.ranges)[::2])
                timestamps.append(t.to_sec())
            elif topic == 'Ackermann':
                raw_servo.append(msg.drive.steering_angle)
                raw_speed.append(msg.drive.speed)
        bag.close()

    raw_speed = np.array(raw_speed)
    min_s, max_s = raw_speed.min(), raw_speed.max()
    raw_speed = linear_map(raw_speed, min_s, max_s, 0, 1)
    raw_servo = np.array(raw_servo)
    timestamps = np.array(timestamps)

    X, y = create_lidar_sequences(raw_lidar, raw_servo, raw_speed, timestamps, seq_len)
    num_samples, _, num_ranges = X.shape
    print(f"Total sequences: {num_samples}, num_ranges: {num_ranges}")

    X, y = shuffle(X, y, random_state=62)
    split = int(0.85 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = X_train.reshape(-1, seq_len, num_ranges, 1)
    X_test  = X_test.reshape(-1, seq_len, num_ranges, 1)

    model = build_spatiotemporal_model(seq_len, num_ranges)
    model.compile(optimizer=Adam(lr), loss='huber')
    print(model.summary())

    for layer in model.layers:
        if 'lstm' in layer.name or 'attention' in layer.name:
            layer.trainable = False
    model.fit(X_train, y_train, epochs=5, batch_size=batch_size,


# how to train
#         1.for layer in model.layers:
#             if 'lstm' in layer.name or 'attention' in layer.name:
#                 layer.trainable = False
#   • Train on X_train (shape: [N, 5, num_ranges, 1]) to learn spatial features
#
# 2.3 frame window
#         for layer in model.layers:
#             layer.trainable = True
#   train on X_train[:, :3] (shape: [N, 3, num_ranges, 1]) for LSTM+Attention 


#then
#model.fit(X_train, y_train, …)