#!/usr/bin/env python3
import argparse
import os
import time
import glob
import csv

import numpy as np
import tensorflow as tf
# --- at top of file ---
import tensorflow as tf

# inside benchmark_tflite():
gpu_delegate = tf.lite.experimental.load_delegate(
    "libtensorflowlite_gpu_delegate.so"
)
interpreter = tf.lite.Interpreter(
    model_path=model_path,
    experimental_delegates=[gpu_delegate],
    # num_threads is ignored by GPU delegate
)
interpreter.allocate_tensors()

def find_trt_delegate():
    # 1) Try the well‐known filenames
    for name in ("libnvinfer_delegate.so",
                 "libtensorflowlite_tensorrt_delegate.so"):
        try:
            tf.lite.experimental.load_delegate(name)
            return name
        except OSError:
            pass

    # 2) Search inside the TF pip package
    tf_pkg = os.path.dirname(tf.__file__)
    # look under .../lite or .../tflite for any delegate .so
    pattern = os.path.join(tf_pkg, "**", "*delegate*.so*")
    candidates = glob.glob(pattern, recursive=True)
    if candidates:
        # pick the first matching library
        return candidates[0]

    raise OSError("TensorRT delegate .so not found in TF install")

def load_csv_sequences(csv_path, seq_len):
    # Read header
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    ts_i = header.index("timestamp")
    rng_idxs = [i for i,c in enumerate(header) if c.startswith("range_")]

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                      usecols=[ts_i] + rng_idxs, dtype=np.float32)
    ts = data[:,0]
    lidar = data[:,1:]

    seqs = []
    for i in range(len(lidar)-seq_len):
        frames = lidar[i:i+seq_len]
        dt = np.diff(ts[i:i+seq_len+1]).reshape(seq_len,1)
        dt = np.repeat(dt, frames.shape[1], axis=1)
        seqs.append(np.stack([frames,dt],axis=-1))
    return np.array(seqs)

def benchmark_tflite(model_path, sequences):
    lib = find_trt_delegate()
    print(f"Using TRT delegate from: {lib}")
    delegate = tf.lite.experimental.load_delegate(lib)
    interp = tf.lite.Interpreter(model_path=model_path,
                                 experimental_delegates=[delegate],
                                 num_threads=1)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    # warm-up
    for _ in range(5):
        interp.set_tensor(inp["index"], sequences[:1].astype(inp["dtype"]))
        interp.invoke()

    times = []
    for seq in sequences:
        x = seq[None,...].astype(inp["dtype"])
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        _ = interp.get_tensor(out["index"])
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    times = np.array(times)

    print(f"TFLite+TRT-delegate ({os.path.basename(model_path)}):")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  90th%: {np.percentile(times,90):.2f} ms\n")

def benchmark_savedmodel(model_dir, sequences):
    model = tf.saved_model.load(model_dir)
    infer = model.signatures["serving_default"]

    # warm-up
    dummy = tf.constant(sequences[:1].astype(np.float32))
    for _ in range(5):
        infer(lidar_sequence=dummy)

    times = []
    for seq in sequences:
        x = tf.constant(seq[None,...].astype(np.float32))
        t0 = time.perf_counter()
        infer(lidar_sequence=x)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    times = np.array(times)

    print(f"SavedModel ({os.path.basename(model_dir)}):")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  90th%: {np.percentile(times,90):.2f} ms\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help=".tflite file or SavedModel dir")
    p.add_argument("--csv",   required=True,
                   help="CSV with timestamp & range_* cols")
    p.add_argument("--seq_len", type=int, default=5)
    args = p.parse_args()

    seqs = load_csv_sequences(args.csv, args.seq_len)
    print(f"Loaded {len(seqs)} sequences of shape {seqs.shape[1:]}")

    if args.model.endswith(".tflite"):
        benchmark_tflite(args.model, seqs)
    elif os.path.isdir(args.model):
        benchmark_savedmodel(args.model, seqs)
    else:
        print("Error: model must be .tflite or a SavedModel directory.")

if __name__ == "__main__":
    main()
