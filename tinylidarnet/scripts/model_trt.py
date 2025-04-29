# #!/usr/bin/env python3
# import os
# # point XLA at libdevice, disable XLA JIT
# os.environ['XLA_FLAGS']    = '--xla_gpu_cuda_data_dir=' + os.environ.get('CUDA_HOME','') + '/nvvm/libdevice'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# import tensorflow as tf
# tf.config.optimizer.set_jit(False)

# import time, glob, argparse
# import numpy as np
# import pandas as pd
# from tensorflow.python.compiler.tensorrt import trt_convert as trt
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from sklearn.utils import shuffle

# # GPU memory growth
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     for g in gpus:
#         tf.config.experimental.set_memory_growth(g, True)
#     print("GPUs detected:", gpus)
# else:
#     print("No GPU detected.")

# def linear_map(x, x_min, x_max, y_min, y_max):
#     return (x-x_min)/(x_max-x_min)*(y_max-y_min)+y_min

# def read_csv_data(csv_path):
#     df = pd.read_csv(csv_path)
#     ts       = df['timestamp'].astype(np.float64).values
#     steering = df['steering'].astype(np.float32).values
#     speed    = df['speed'].astype(np.float32).values
#     ranges   = df.filter(like='range_').astype(np.float32).values
#     return ranges, steering, speed, ts

# def create_lidar_sequences(lidar, servo, speed, ts, seq_len=5):
#     X, y = [], []
#     nr = lidar.shape[1]
#     for i in range(len(lidar)-seq_len):
#         frames = lidar[i:i+seq_len]
#         dt     = np.diff(ts[i:i+seq_len+1]).reshape(seq_len,1)
#         dt_t   = np.repeat(dt, nr, axis=1)
#         seq    = np.concatenate([frames[...,None], dt_t[...,None]], axis=2)
#         X.append(seq)
#         y.append([servo[i+seq_len], speed[i+seq_len]])
#     return np.array(X), np.array(y)

# from tensorflow.keras.layers import (
#     Input, TimeDistributed, Conv1D, Flatten,
#     LSTM, Dense, Attention, GlobalAveragePooling1D
# )
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# def build_spatiotemporal_model(seq_len, num_ranges):
#     inp = Input((seq_len, num_ranges, 2), name='lidar_sequence')
#     x = TimeDistributed(Conv1D(24,10,strides=4,activation='relu'))(inp)
#     x = TimeDistributed(Conv1D(36,8,strides=4,activation='relu'))(x)
#     x = TimeDistributed(Conv1D(48,4,strides=2,activation='relu'))(x)
#     x = TimeDistributed(Flatten())(x)
#     # use built-in LSTM with unroll=True to avoid dynamic TensorList
#     x = LSTM(64, return_sequences=True, unroll=True)(x)
#     x = LSTM(64, return_sequences=True, unroll=True, go_backwards=True)(x)
#     # now x is (None, seq_len, 64); you could concatenate forward+back if desired
#     # attention
#     q = Dense(64)(x)
#     k = Dense(64)(x)
#     v = Dense(64)(x)
#     attn = Attention()([q,v,k])                  # (None, seq_len, 64)
#     context = GlobalAveragePooling1D()(attn)     # (None, 64)
#     out = Dense(2, activation='tanh', name='controls')(context)
#     return Model(inp, out, name='RNN_Attn_Controller')

# if __name__=='__main__':
#     p = argparse.ArgumentParser()
#     p.add_argument('--csv',  type=str)
#     p.add_argument('--csvs', nargs='+')
#     args = p.parse_args()

#     if args.csv:   paths = [args.csv]
#     elif args.csvs: paths = args.csvs
#     else:
#         base = os.path.dirname(__file__)
#         paths = glob.glob(base+'/sim_Dataset/**/*.csv', recursive=True)
#         if not paths: raise FileNotFoundError("No CSVs found.")

#     seq_len, batch_size, lr, epochs = 5, 16, 5e-5, 20

#     all_l, all_s, all_sp, all_t = [], [], [], []
#     for f in paths:
#         print("Loading", f)
#         l,s,sp,ts = read_csv_data(f)
#         all_l.append(l); all_s.append(s); all_sp.append(sp); all_t.append(ts)
#     lidar = np.vstack(all_l)
#     servo = np.concatenate(all_s)
#     speed= np.concatenate(all_sp)
#     ts   = np.concatenate(all_t)

#     mn, mx = speed.min(), speed.max()
#     speed = linear_map(speed,mn,mx,0,1) if mx>mn else np.zeros_like(speed)

#     X, y = create_lidar_sequences(lidar,servo,speed,ts,seq_len)
#     print("Sequences:", X.shape, "Targets:", y.shape)
#     X, y = shuffle(X,y,random_state=42)
#     sp = int(0.85*len(X))
#     Xtr, Xte = X[:sp], X[sp:]
#     ytr, yte = y[:sp], y[sp:]

#     model = build_spatiotemporal_model(seq_len, lidar.shape[1])
#     model.compile(Adam(lr), loss='huber')
#     model.summary()

#     history = model.fit(Xtr,ytr, validation_data=(Xte,yte),
#                         epochs=epochs, batch_size=batch_size)
#     print("Training complete.")

#     # plot
#     plt.plot(history.history['loss'], label='train')
#     plt.plot(history.history['val_loss'], label='val')
#     plt.legend(); plt.xlabel('epoch'); plt.ylabel('loss')
#     os.makedirs('Figures',exist_ok=True)
#     plt.savefig('Figures/loss_curve.png')

#     # save Keras .keras
#     os.makedirs('Models',exist_ok=True)
#     kp = 'Models/RNN_Attn_Controller.keras'
#     model.save(kp)
#     print("Saved Keras →", kp)

#     # export SavedModel
#     sm = 'Models/RNN_Attn_Controller_SavedModel'
#     model.export(sm)
#     print("Exported SavedModel →", sm)

#     # TFLite
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.target_spec.supported_ops = [
#         tf.lite.OpsSet.TFLITE_BUILTINS,
#         tf.lite.OpsSet.SELECT_TF_OPS
#     ]
#     tfl = converter.convert()
#     with open('Models/RNN_Attn_Controller.tflite','wb') as f: f.write(tfl)
#     print("Saved TFLite model")

#     # TF-TRT (guard if not built-with-TRT)
#     try:
#         trt_params = trt.TrtConversionParams(
#             precision_mode=trt.TrtPrecisionMode.FP16,
#             max_workspace_size_bytes=(1<<28)
#         )
#         trt_conv = trt.TrtGraphConverterV2(
#             input_saved_model_dir=sm,
#             conversion_params=trt_params
#         )
#         trt_conv.convert()
#         trt_conv.save('Models/RNN_Attn_Controller_TrtFP16')
#         print("Saved TF-TRT FP16 model")
#     except Exception as e:
#         print("Skipping TF-TRT conversion:", e)

#     loss = model.evaluate(Xte,yte,verbose=0)
#     print("Final test loss:", loss)


import tensorflow as tf
import numpy as np
import pandas as pd

# 1) Your calibration generator must return a list of input tensors:
def calibration_input_fn():
    df = pd.read_csv("test_levine3_0.csv")
    seq_len = 5
    ranges = [c for c in df.columns if c.startswith("range_")]
    ts    = df["timestamp"].values.astype(np.float32)
    lidar = df[ranges].values.astype(np.float32)

    for i in range(len(lidar) - seq_len):
        frames = lidar[i : i + seq_len]
        dt     = np.diff(ts[i : i + seq_len + 1]).astype(np.float32).reshape(seq_len,1)
        dt     = np.repeat(dt, frames.shape[1], axis=1)
        seq    = np.stack([frames, dt], axis=-1)
        # MUST yield a Python list matching your SavedModel signature
        yield [seq]

# 2) Create the converter in INT8 mode
converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir="Models/RNN_Attn_Controller_SavedModel",
    conversion_params=tf.experimental.tensorrt.ConversionParams(
        precision_mode="INT8",
        max_workspace_size_bytes=1 << 30,
        use_calibration=True     # enable calibration
    )
)

# 3) Run the generic graph conversion
converter.convert()

# 4) **Now** run calibration on a handful of samples
converter.build(input_fn=calibration_input_fn)

# 5) Save your newly‐calibrated INT8 TF-TRT SavedModel
converter.save("Models/RNN_Attn_Controller_TFTRT_INT8")
