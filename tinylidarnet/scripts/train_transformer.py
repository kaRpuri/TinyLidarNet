#!/usr/bin/env python3
"""
train_transformer_lightweight.py

Single‐block, small‐dimension Transformer for LiDAR→controls:
- seq_len=5
- d_model=128, num_heads=4, ff_dim=256
- single pre‑norm Transformer block
- sinusoidal positional encoding
- GlobalAveragePooling1D pooling
- no CLS token, no auxiliary head
- warmup+cosine learning rate schedule, gradient clipping
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# ROS 2 bag imports
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


# Keras imports
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    BatchNormalization, Dropout,
    Dense, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#========================================================
# Utility & data functions
#========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def read_ros2_bag(bag_path):
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions('', '')
    reader = SequentialReader(); reader.open(storage_opts, conv_opts)

    lidar, servo, speed, ts = [], [], [], []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t = t_ns * 1e-9
        if topic == 'scan':
            msg = deserialize_message(data, LaserScan)
            cleaned = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar.append(cleaned[::2])  # subsample by 2
            ts.append(t)
        elif topic == 'odom':
            msg = deserialize_message(data, Odometry)
            servo.append(msg.twist.twist.angular.z)
            speed.append(msg.twist.twist.linear.x)
    return np.array(lidar), np.array(servo), np.array(speed), np.array(ts)

def create_lidar_sequences(lidar, servo, speed, ts, seq_len=5):
    X, y = [], []
    num_ranges = lidar.shape[1]
    for i in range(len(lidar) - seq_len):
        frames = np.stack(lidar[i:i+seq_len], axis=0)      # (seq_len, num_ranges)
        dt     = np.diff(ts[i:i+seq_len+1]).reshape(seq_len,1)
        dt_tiled = np.repeat(dt, num_ranges, axis=1)
        seq    = np.concatenate([frames[...,None], dt_tiled[...,None]], axis=2)
        X.append(seq)
        y.append([servo[i+seq_len], speed[i+seq_len]])
    return np.array(X), np.array(y)

#========================================================
# Positional encoding
#========================================================
def sinusoidal_pos_enc(seq_len, d_model):
    pos = np.arange(seq_len)[:,None]
    i   = np.arange(d_model)[None,:]
    angle = pos / np.power(10000, (2*(i//2))/d_model)
    pe = np.zeros((seq_len,d_model))
    pe[:,0::2] = np.sin(angle[:,0::2])
    pe[:,1::2] = np.cos(angle[:,1::2])
    return tf.constant(pe, dtype=tf.float32)

#========================================================
# Single‐block Transformer
#========================================================
def build_transformer_model(seq_len, num_ranges,
                            d_model=128, num_heads=4, ff_dim=256,
                            dropout_rate=0.1):
    inp = Input((seq_len, num_ranges, 2), name='lidar_seq')

    # per‑frame conv backbone
    x = TimeDistributed(Conv1D(24,10,strides=4,activation='relu'))(inp)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv1D(36,8,strides=4,activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Flatten())(x)  # (batch, seq_len, feat_dim)

    # project to d_model
    x = TimeDistributed(Dense(d_model))(x)

    # add sinusoidal pos enc
    pe = sinusoidal_pos_enc(seq_len, d_model)[None,...]
    x = x + pe

    # single pre‑norm Transformer block
    y = LayerNormalization(epsilon=1e-6)(x)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(y,y,y)
    attn = Dropout(dropout_rate)(attn)
    x2 = x + attn

    y2 = LayerNormalization(epsilon=1e-6)(x2)
    ff = Dense(ff_dim, activation='relu')(y2)
    ff = Dense(d_model)(ff)
    ff = Dropout(dropout_rate)(ff)
    x3 = x2 + ff

    # pool & head
    context = GlobalAveragePooling1D()(x3)
    out     = Dense(2, activation='tanh', name='controls')(context)

    return Model(inp, out, name='LiteTransformer')

#========================================================
# LR schedule: warmup+cosine
#========================================================
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base = base_lr
        self.warm = tf.cast(warmup_steps, tf.float32)
        self.total= tf.cast(total_steps, tf.float32)
    def __call__(self, step):
        s = tf.cast(step, tf.float32)
        warm_lr = self.base * (s/self.warm)
        prog = (s - self.warm)/(self.total - self.warm)
        cos_lr = 0.5*self.base*(1+tf.cos(np.pi*prog))
        return tf.where(s<self.warm, warm_lr, cos_lr)

#========================================================
# Main
#========================================================
if __name__=='__main__':
    print("GPU AVAILABLE:", tf.config.list_physical_devices('GPU'))

    # paths + hyperparams
    bag_paths = [
    '/home/shirin/lab_ws/TinyLidarNet/tinylidarnet/scripts/sim_Dataset/test_levine1/test_levine1_0.db3']

    seq_len, batch_size, epochs = 5, 128, 20

    # load all bags
    all_lidar, all_servo, all_speed, all_ts = [],[],[],[]
    for p in bag_paths:
        l,s,sp,ts = read_ros2_bag(p)
        print(f"Loaded {len(l)} scans from {p}")
        all_lidar.extend(l); all_servo.extend(s)
        all_speed.extend(sp); all_ts.extend(ts)
    all_lidar = np.array(all_lidar)
    all_servo = np.array(all_servo)
    all_speed = np.array(all_speed)
    all_ts    = np.array(all_ts)

    # normalize speed
    vmin, vmax = all_speed.min(), all_speed.max()
    all_speed  = linear_map(all_speed, vmin, vmax, 0, 1)

    # build sequences
    X, y      = create_lidar_sequences(all_lidar, all_servo, all_speed, all_ts, seq_len)
    n,_,nr,_  = X.shape
    print(f"Total sequences: {n}, ranges/frame: {nr}")

    # shuffle & split
    X, y = shuffle(X,y, random_state=42)
    split = int(0.85 * n)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    # build & compile
    model = build_transformer_model(seq_len, nr)
    total_steps = (n//batch_size)*epochs
    lr_sched    = WarmupCosine(2e-4, warmup_steps=200, total_steps=total_steps)
    opt = Adam(learning_rate=lr_sched, clipnorm=1.0)
    model.compile(optimizer=opt, loss='huber')
    model.summary()

    # callbacks
    callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]

    # train
    t0 = time.time()
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xte,yte),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    print(f"Training completed in {time.time()-t0:.1f}s")

    # plot & save
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig('loss_curve_lite.png')

    # export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    os.makedirs('Models', exist_ok=True)
    open('Models/lite_transformer.tflite','wb').write(tflite)
    print("Saved lite_transformer.tflite")
