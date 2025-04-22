#!/usr/bin/env python3
"""
train_transformer.py

Advanced Transformer-based spatio-temporal LiDAR→controls training:
- Sinusoidal positional embeddings
- [CLS] token pooling (custom Keras layer)
- Stacked Transformer encoder blocks (pre-norm)
- Warmup + cosine decay learning rate schedule
- Gradient clipping, dropout, LayerNorm
- Auxiliary time-to-collision head
"""

import os
import time
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

# Keras imports
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    BatchNormalization, Dropout,
    Dense, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Embedding, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#========================================================
# Positional encoding utilities
#========================================================
def sinusoidal_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i   = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i//2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(pe, dtype=tf.float32)

#========================================================
# Custom [CLS] token layer
#========================================================
class CLSToken(Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        # input_shape: (batch, seq_len, d_model)
        self.cls = self.add_weight(
            shape=(1, 1, self.d_model),
            initializer='zeros',
            trainable=True,
            name='cls_token'
        )
    def call(self, inputs):
        # inputs: (batch, seq_len, d_model)
        batch = tf.shape(inputs)[0]
        return tf.tile(self.cls, [batch, 1, 1])

#========================================================
# Transformer block (pre-norm)
#========================================================
def transformer_block(x, d_model, num_heads, ff_dim, dropout_rate=0.1):
    # Pre-norm
    y    = LayerNormalization(epsilon=1e-6)(x)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(y, y, y)
    attn = Dropout(dropout_rate)(attn)
    x1   = x + attn

    y1  = LayerNormalization(epsilon=1e-6)(x1)
    ff  = Dense(ff_dim, activation='relu')(y1)
    ff  = Dense(d_model)(ff)
    ff  = Dropout(dropout_rate)(ff)
    x2  = x1 + ff
    return x2

#========================================================
# Read and build sequences
#========================================================
def read_ros2_bag(bag_path):
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader(); reader.open(storage_opts, conv_opts)
    lidar, servo, speed, ts = [], [], [], []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t = t_ns * 1e-9
        if topic == 'scan':
            msg = deserialize_message(data, LaserScan)
            cleaned = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar.append(cleaned[::2]); ts.append(t)
        elif topic == 'odom':
            msg = deserialize_message(data, Odometry)
            servo.append(msg.twist.twist.angular.z)
            speed.append(msg.twist.twist.linear.x)
    return np.array(lidar), np.array(servo), np.array(speed), np.array(ts)

def create_lidar_sequences(lidar, servo, speed, ts, seq_len=5):
    X, y, y_ttc = [], [], []
    num_ranges = lidar.shape[1]
    for i in range(len(lidar)-seq_len):
        frames = np.stack(lidar[i:i+seq_len], axis=0)
        dt = np.diff(ts[i:i+seq_len+1]).reshape(seq_len,1)
        dt_tiled = np.repeat(dt, num_ranges, axis=1)
        seq = np.concatenate([frames[...,None], dt_tiled[...,None]], axis=2)
        X.append(seq)
        y.append([servo[i+seq_len], speed[i+seq_len]])
        y_ttc.append(np.min(frames[-1]))
    return np.array(X), np.array(y), np.array(y_ttc)

#========================================================
# Model builder
#========================================================
def build_advanced_transformer(seq_len, num_ranges,
                               d_model=256, num_heads=8,
                               ff_dim=512, num_layers=4,
                               dropout_rate=0.1, aux_ttc=False):
    inp = Input((seq_len, num_ranges, 2), name='lidar_seq')
    # Conv backbone per frame
    x = TimeDistributed(Conv1D(32,10, strides=4, activation='relu'))(inp)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(Conv1D(64,8, strides=4, activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)
    x = TimeDistributed(Conv1D(128,4, strides=2, activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Flatten())(x)  # (batch, seq_len, feat_dim)

    # Project to d_model
    proj = TimeDistributed(Dense(d_model))(x)

    # CLS token layer
    cls_tok = CLSToken(d_model)(proj)

    # Positional encodings
    pe = sinusoidal_positional_encoding(seq_len, d_model)[None,...]
    pos = proj + pe

    # Concat CLS + pos
    x2 = tf.keras.layers.Concatenate(axis=1)([cls_tok, pos])  # (batch, seq_len+1, d_model)

    # Stacked transformer blocks
    for _ in range(num_layers):
        x2 = transformer_block(x2, d_model, num_heads, ff_dim, dropout_rate)

    # CLS output
    cls_out = x2[:,0,:]
    ctrl    = Dense(2, activation='tanh', name='controls')(cls_out)

    if aux_ttc:
        ttc = Dense(1, activation='relu', name='time_to_collision')(cls_out)
        return Model(inp, [ctrl, ttc], name='AdvTransformer')

    return Model(inp, ctrl, name='AdvTransformer')

class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base = base_lr
        self.warmup = tf.cast(warmup_steps, tf.float32)
        self.total = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        # cast step to float32
        s = tf.cast(step, tf.float32)
        # linear warmup: base_lr * (s / warmup)
        warmup_lr = self.base * (s / self.warmup)
        # cosine decay: 0.5*base*(1 + cos(pi * (s - warmup)/(total-warmup)))
        progress = (s - self.warmup) / (self.total - self.warmup)
        cosine_lr = 0.5 * self.base * (1 + tf.cos(np.pi * progress))
        # if s < warmup → warmup_lr else → cosine_lr
        return tf.where(s < self.warmup, warmup_lr, cosine_lr)


#========================================================
# Main training routine
#========================================================
if __name__=='__main__':
    print("GPU AVAILABLE:", tf.config.list_physical_devices('GPU'))

    bag_paths = [
        '/home/shirin/lab_ws/TinyLidarNet/tinylidarnet/scripts/sim_Dataset/test_levine1/test_levine1_0.db3'
    ]
    seq_len, batch_size, epochs = 5, 64, 30

    lidar, servo, speed, ts = [],[],[],[]
    for p in bag_paths:
        l,s,sp,t = read_ros2_bag(p)
        lidar.extend(l); servo.extend(s); speed.extend(sp); ts.extend(t)
    lidar, servo, speed, ts = map(np.array, (lidar, servo, speed, ts))

    # Normalize speed
    vmin, vmax = speed.min(), speed.max()
    speed = (speed-vmin)/(vmax-vmin)

    # Build sequences
    X, y, y_ttc = create_lidar_sequences(lidar, servo, speed, ts, seq_len)
    n = X.shape[0]
    X, y, y_ttc = shuffle(X,y,y_ttc, random_state=42)
    split = int(0.85*n)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    ttctr, ttcte = y_ttc[:split], y_ttc[split:]

    # Instantiate model
    model = build_advanced_transformer(seq_len, X.shape[2], aux_ttc=True)
    total_steps = (n//batch_size)*epochs
    lr_sched = WarmupCosine(1e-4, 500, total_steps)
    opt = Adam(learning_rate=lr_sched, clipnorm=1.0)
    model.compile(
        optimizer=opt,
        loss={'controls':'huber','time_to_collision':'mse'},
        loss_weights={'controls':1.0,'time_to_collision':0.5}
    )
    model.summary()

    cb = [EarlyStopping(patience=5, restore_best_weights=True)]
    hist = model.fit(
        Xtr, {'controls':ytr,'time_to_collision':ttctr},
        validation_data=(Xte,{'controls':yte,'time_to_collision':ttcte}),
        epochs=epochs, batch_size=batch_size, callbacks=cb
    )
    print("Training time:", time.time()-t0)

    plt.plot(hist.history['loss'],label='train')
    plt.plot(hist.history['val_loss'],label='val')
    plt.legend(); plt.savefig('loss_curve.png')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    open('adv_transformer.tflite','wb').write(tflite_model)
    print("Saved TFLite model")