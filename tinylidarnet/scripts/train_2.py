#!/usr/bin/env python3
import os, time, warnings, numpy as np, tensorflow as tf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from sklearn.utils import shuffle   # still imported even if not used

print("GPU AVAILABLE:", bool(tf.config.list_physical_devices("GPU")))

# ───────────────────────────────────────── utils
def linear_map(x, x_min, x_max, y_min, y_max):
    if x_max == x_min:                      # avoid divide‑by‑zero
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

# ───────────────────────────────────────── bag reader
def read_ros2_bag(bag_path):
    storage = StorageOptions(uri=bag_path, storage_id="sqlite3")
    reader = SequentialReader(); reader.open(storage, ConverterOptions('', ''))
    lidar, servo, speed = [], [], []

    while reader.has_next():
        topic, data, _ = reader.read_next()

        if topic.endswith('/scan'):         # robust match
            msg = deserialize_message(data, LaserScan)
            lidar.append(np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)[::2])

        elif topic in {'/cmd_vel'}:
            msg = deserialize_message(data, Twist)
            servo.append(msg.angular.z);  speed.append(msg.linear.x)

        elif topic in {'/drive','/ackermann_cmd'}:
            msg = deserialize_message(data, AckermannDriveStamped)
            servo.append(msg.drive.steering_angle);  speed.append(msg.drive.speed)

    return np.asarray(lidar), np.asarray(servo), np.asarray(speed)

# ───────────────────────────────────────── main
if __name__ == "__main__":
    BAG = "/home/nvidia/f1tenth_ws/src/TinyLidarNet/tinylidarnet/scripts/sim_Dataset/test_kabir/testrun3_0.db3"
    LR, BATCH, EPOCHS = 5e-5, 64, 20

    l_data, s_data, sp_data = read_ros2_bag(BAG)
    idx = np.random.permutation(len(l_data)); split = int(0.85 * len(idx))

    train_lid = l_data[idx[:split]][..., None]
    test_lid  = l_data[idx[split:]][..., None]

    min_sp, max_sp = sp_data.min(), sp_data.max()
    train_sp = linear_map(sp_data[idx[:split]], min_sp, max_sp, 0, 1)
    test_sp  = linear_map(sp_data[idx[split:]], min_sp, max_sp, 0, 1)

    train_sv, test_sv = s_data[idx[:split]], s_data[idx[split:]]

    print(f"Dataset  train:{len(train_lid)}  test:{len(test_lid)}  "
          f"speed {min_sp:.2f}–{max_sp:.2f}")

    # — model (unchanged except Input layer added for clarity)
    N = train_lid.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input((N,1)),
        tf.keras.layers.Conv1D(24,10,4,activation='relu'),
        tf.keras.layers.Conv1D(36,8,4,activation='relu'),
        tf.keras.layers.Conv1D(48,4,2,activation='relu'),
        tf.keras.layers.Conv1D(64,3,activation='relu'),
        tf.keras.layers.Conv1D(64,3,activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100,activation='relu'),
        tf.keras.layers.Dense(50,activation='relu'),
        tf.keras.layers.Dense(10,activation='relu'),
        tf.keras.layers.Dense(2,activation='tanh')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='huber')
    model.summary()

    t0 = time.time()
    hist = model.fit(
        train_lid, np.stack([train_sv, train_sp],1),
        validation_data=(test_lid, np.stack([test_sv, test_sp],1)),
        epochs=EPOCHS, batch_size=BATCH, verbose=2)
    print("Training time:", int(time.time()-t0), "s")

    # — loss curve
    os.makedirs("Figures",exist_ok=True)
    plt.plot(hist.history['loss']), plt.plot(hist.history['val_loss'])
    plt.legend(['train','val']); plt.xlabel('epoch'); plt.ylabel('loss')
    plt.title('TLN loss'); plt.savefig('Figures/loss_curve.png'); plt.close()

    # — TFLite export (legacy converter, Flex fallback)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_use_mlir_converter = False   # ← key line
    tflite_model = converter.convert()

    os.makedirs("Models",exist_ok=True)
    open("Models/TLN_noquantized.tflite","wb").write(tflite_model)
    print("✓ exported  Models/TLN_noquantized.tflite")

    print("Final val‑loss:", model.evaluate(test_lid, np.stack([test_sv,test_sp],1),verbose=0))
