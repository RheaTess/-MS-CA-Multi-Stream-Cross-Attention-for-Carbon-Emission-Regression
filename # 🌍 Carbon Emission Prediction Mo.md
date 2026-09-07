# 🌍 MS-CA Carbon Emission Prediction Model (Pure Regression)

## 📌 Project Overview
This project is a deep learning regression pipeline that uses satellite proxy data to predict the continuous carbon emission value (a continuous scalar) for over 17,000 regional cells. We have completely discarded the previous hotspot classification approach. Our new objective is to preserve the outliers in the raw data and objectively demonstrate the true correlation between the proxy features and the actual carbon map.

---

## 🏗️ Data Flow & Model Architecture (Tensor Shape Transformation)
Below is the 5-step computational process detailing how the model reads a 16x16 pixel environment around a cell and ultimately outputs exactly 1 scalar prediction (carbon emission).

* **Notation:** $B$ = Batch Size, $C$ = Channels, $H$/$W$ = Image Height/Width, $D$ = Embedding Dimension.

### Step 1. Data Input & Crop
The model extracts a 16x16 pixel spatial window centered on a specific cell.
* **Stream A (Environment Data):** A tensor of shape `[B, 3, 16, 16]` consisting of NO2, SO2, and CO.
* **Stream B (Infrastructure Data):** A tensor of shape `[B, 4, 16, 16]` consisting of Nightlight, Urban Fraction, Power Plant, and Fossil Capacity.

### Step 2. Patch Partition & Embedding (ViT Tokenization)
The Vision Transformer (ViT) encoder splits the 16x16 image into sixteen 4x4 patches.
* Each patch (a block of pixels) is converted into an information vector (embedding) consisting of 128 numbers.
* **Current Shape:** `[B, 16, 128]` (16 patches, each carrying 128 dimensions of information).

### Step 3. Cross-Attention Fusion
The information from the two streams is combined via Cross-Attention.
* Using Stream B (Infrastructure) as the Query and Stream A (Environment) as the Key/Value, the model learns which environmental patches to focus on based on the infrastructural context.
* **Current Shape:** `[B, 16, 128]` (The number of patches and dimensions remain unchanged after fusion).

### Step 4. Global Average Pooling (Information Compression)
The 16 separate patch vectors are averaged (Mean) and compressed into a single vector.
* This creates a single summary representation that describes the overall characteristics of the neighborhood around the cell.
* **Current Shape:** `[B, 128]` (16 patches compressed into 1 vector).

### Step 5. Regression Head (Final Output)
The compressed 128-dimensional summary vector passes through a Multi-Layer Perceptron (MLP) to be converted into the final numerical value.
* **Computation Flow:** 128 dimensions -> 64 dimensions -> 1 dimension.
* **Final Shape:** `[B, 1]`.
* **Result:** Through this pipeline, exactly one accurate carbon emission scalar value is computed for each of the 17,000 cells.

---

## 🚀 Pipeline Execution Guide

### 1. Training (`train.py`)
This script starts model training and saves the model when the validation error is at its lowest. The evaluation is based on **MAE (Mean Absolute Error, L1 Loss)**.

```bash
# Basic Execution Command
python train.py \
    --data_path ./data/month_data.npz \
    --output_dir ./checkpoints \
    --epochs 50 \
    --batch_size 32 \
    --lr 3e-4 \
    --seed 42

```

### 2. Final Evaluation (evaluation.py)
After training is complete, this script loads the saved best_model.pth to evaluate the unseen test set (15%).

```Bash
# Note: You MUST use the identical --seed, --val_ratio, and --test_ratio as train.py to reconstruct the exact same test split.
python evaluation.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_path ./data/month_data.npz \
    --seed 42
```

## 📊 Final Evaluation Metrics (Test Metrics)
The evaluation on the test set uses the following 4 metrics to objectively prove the correlation between our proxy data and the actual carbon emission map.

MAE (Mean Absolute Error): The absolute average difference between the actual and predicted values.

RMSE (Root Mean Squared Error): Gives a higher penalty to larger errors, acting as an indicator of model stability.

R^2 (Coefficient of Determination): Represents how well the model explains the variance of the actual data (closer to 1 is perfect).

Pearson correlation (r): The key performance indicator showing how linearly correlated our prediction map (made from 7 proxy variables) is with the actual carbon map.


