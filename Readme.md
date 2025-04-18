# Multi-Modal-TinyLidarNet

**Multi-Modal-TinyLidarNet** is a modular, ROS 2-compatible project for multi-modal sensor data processing, benchmarking, and neural network inference, designed for lightweight deployment and experimentation. 

---


---

## ⚙️ Requirements

- **Operating System:** Linux (Ubuntu 22.04 recommended)  
- **Python:** 3.10+  
- **ROS 2:** Humble Hawksbill (or compatible)  
- **Conda:** Required for environment and dependency management  

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/kaRpuri/TinyLidarNet.git
cd TinyLidarNet
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda install pip
which pip   # make sure that you are using the correct pip
conda activate tln
```

### 3. Install ROS 2 (if not already installed)


Source your ROS 2 setup script in every new terminal:


---

## 🚀 Usage

### Data Collection

```bash
cd tinylidarnet/scripts
python data_collection.py
```

### Training

```bash
cd tinylidarnet/scripts
python train.py
```

### Inference

```bash
cd tinylidarnet/scripts
python inference.py
```





