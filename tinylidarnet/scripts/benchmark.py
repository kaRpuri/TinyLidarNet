import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------------
# 1) Load the test CSV into sequences
# -----------------------------------
df = pd.read_csv("test_levine3_0.csv")
ts       = df['timestamp'].values.astype(np.float64)
steer_gt = df['steering'].values.astype(np.float32)
speed_gt = df['speed'].values.astype(np.float32)
ranges   = [c for c in df if c.startswith('range_')]
lidar    = df[ranges].values.astype(np.float32)

seq_len, r = 5, len(ranges)
sequences = []
labels    = []
for i in range(len(lidar) - seq_len):
    frames = lidar[i:i+seq_len]                                   # (5, r)
    dt     = np.diff(ts[i:i+seq_len+1]).reshape(seq_len,1)        # (5,1)
    dt     = np.repeat(dt, r, axis=1)                             # (5,r)
    seq    = np.stack([frames, dt], axis=-1).astype(np.float32)   # (5,r,2)
    sequences.append(seq)
    labels.append((steer_gt[i+seq_len], speed_gt[i+seq_len]))

# -----------------------------------
# 2) Helper to compute MSE/MAE + latency
# -----------------------------------
def compute_metrics(gts, preds, times_ms):
    gts   = np.array(gts)
    preds = np.array(preds)
    return {
        "MSE Steering":    mean_squared_error(gts[:,0], preds[:,0]),
        "MAE Steering":    mean_absolute_error(gts[:,0], preds[:,0]),
        "MSE Speed":       mean_squared_error(gts[:,1], preds[:,1]),
        "MAE Speed":       mean_absolute_error(gts[:,1], preds[:,1]),
        "Latency Mean (ms)": np.mean(times_ms),
        "Latency 90th (ms)": np.percentile(times_ms, 90)
    }

# -----------------------------------
# 3) Run a TFLite model (CPU only)
# -----------------------------------
def benchmark_tflite(model_path, dtype=np.float32):
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    i_idx = interp.get_input_details()[0]['index']
    o_idx = interp.get_output_details()[0]['index']

    # warm-up
    dummy = sequences[0][None].astype(dtype)
    interp.set_tensor(i_idx, dummy)
    interp.invoke()

    preds, times = [], []
    for seq in sequences:
        inp = seq[None].astype(dtype)
        interp.set_tensor(i_idx, inp)
        t0 = time.perf_counter()
        interp.invoke()
        t1 = time.perf_counter()
        out = interp.get_tensor(o_idx)[0]
        preds.append((float(out[0]), float(out[1])))
        times.append((t1 - t0) * 1e3)
    return preds, times

tflite_fp32_preds, tflite_fp32_times = benchmark_tflite("Models/RNN_Attn_Controller.tflite", np.float32)
# tflite_int8_preds, tflite_int8_times = benchmark_tflite("Models/controller_int8.tflite", np.uint8)

# -----------------------------------
# 4) Run the TF-TRT FP16 SavedModel on GPU
# -----------------------------------
trt_model = tf.saved_model.load("Models/RNN_Attn_Controller_TFTRT_FP16")
infer     = trt_model.signatures['serving_default']
inp_name  = list(infer.structured_input_signature[1].keys())[0]
out_name  = list(infer.structured_outputs.keys())[0]

# warm-up
_ = infer(**{inp_name: tf.constant(sequences[0:1], dtype=tf.float32)})

trt_preds, trt_times = [], []
for seq in sequences:
    tensor_in = tf.constant(seq[None], dtype=tf.float32)
    t0 = time.perf_counter()
    out = infer(**{inp_name: tensor_in})[out_name].numpy()[0]
    t1 = time.perf_counter()
    trt_preds.append((float(out[0]), float(out[1])))
    trt_times.append((t1 - t0) * 1e3)

# -----------------------------------
# 5) Run the TensorRT INT8-plan on GPU
# -----------------------------------
plan_path = "model_int8.plan"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime    = trt.Runtime(TRT_LOGGER)
with open(plan_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
context.set_binding_shape(0, (1, seq_len, r, 2))

# Allocate host/device buffers
bindings = [None] * engine.num_bindings
host_in  = []
host_out = []
for b in range(engine.num_bindings):
    shape = context.get_binding_shape(b)
    size  = trt.volume(shape)
    dtype = trt.nptype(engine.get_binding_dtype(b))
    h_mem = cuda.pagelocked_empty(size, dtype)
    d_mem = cuda.mem_alloc(h_mem.nbytes)
    bindings[b] = int(d_mem)
    if engine.binding_is_input(b):
        host_in.append((b, h_mem, d_mem))
    else:
        host_out.append((b, h_mem, d_mem))

stream = cuda.Stream()

# warm-up
host_in[0][1][:] = sequences[0].flatten()
cuda.memcpy_htod_async(host_in[0][2], host_in[0][1], stream)
context.execute_async_v2(bindings, stream.handle)
stream.synchronize()

plan_preds, plan_times = [], []
for seq in sequences:
    # copy in
    host_in[0][1][:] = seq.flatten()
    cuda.memcpy_htod_async(host_in[0][2], host_in[0][1], stream)

    # inference
    t0 = time.perf_counter()
    context.execute_async_v2(bindings, stream.handle)
    stream.synchronize()
    t1 = time.perf_counter()
    plan_times.append((t1 - t0) * 1e3)

    # copy out all output buffers
    for b_idx, h_mem, d_mem in host_out:
        cuda.memcpy_dtoh_async(h_mem, d_mem, stream)
    stream.synchronize()

    # reshape them
    out_arrays = [
        h_mem.reshape(context.get_binding_shape(b_idx))
        for (b_idx, h_mem, d_mem) in host_out
    ]

    # handle single 2-element output or two separate outputs
    if len(out_arrays) == 1:
        arr = out_arrays[0].reshape(-1)
        steering_pred, speed_pred = float(arr[0]), float(arr[1])
    else:
        steering_pred = float(out_arrays[0].flat[0])
        speed_pred    = float(out_arrays[1].flat[0])

    plan_preds.append((steering_pred, speed_pred))

# -----------------------------------
# 6) Collate and print
# -----------------------------------
results = []
for name, (preds, times) in [
    ("TFLite FP32",    (tflite_fp32_preds, tflite_fp32_times)),
    # ("TFLite INT8", (tflite_int8_preds,  tflite_int8_times)),
    ("TF-TRT FP16",    (trt_preds,         trt_times)),
    ("TensorRT INT8", (plan_preds,        plan_times)),
]:
    m = compute_metrics(labels, preds, times)
    m["Model"] = name
    results.append(m)

df = pd.DataFrame(results)
print(df.to_string(index=False))
