# Dynamic Feature Fusion for Blockchain Fraud Detection

This repository implements the **ETH-GBERT** model from the paper *"Dynamic Feature Fusion: Combining Global Graph Structures and Local Semantics for Blockchain Fraud Detection"* by Zhang Sheng, Liangliang Song, and Yanbin Wang ([arXiv:2501.02032v1](https://arxiv.org/abs/2501.02032)).  
The model combines **graph neural networks (GCN)** for transaction structures and **BERT** for semantic features to detect fraud in blockchain data (e.g., Ethereum).

---

## 🔹 Key Features
- Preprocessing pipeline to generate graph & text features (n-gram time differences, adjacency matrix).
- ETH-GBERT model with dynamic multimodal fusion.
- Modifications:
  - GELU activation.
  - Learnable self-loop in GCN.
  - Simplified fusion (g1/g2 gates).
  - Optional Q-Former with cross-attention for enhanced fusion.
- Baselines: Random Forest & SVM for comparison.

---

## 📌 Overview
- **Model Architecture**: GCN captures global transaction graphs; BERT handles local semantics; dynamic fusion for classification.
- **Preprocessing**: Scripts (`dataset1.py` → `BERT_text_data.py`) prepare data from raw transactions.
- **Training**: `train.py` for ETH-GBERT, `train_random_forest.py` & `train_svm.py` for baselines.
- **Evaluation**: Weighted F1, precision, recall (as in the paper, p.9).

---

## ⚙ Requirements
Create a Conda environment and install dependencies:
```bash
conda create -n blockchain_fraud python=3.10
conda activate blockchain_fraud
pip install -r requirements.txt
```

---

## 🚀 Setup & Training

### 1. Clone the Repository
```bash
git clone https://github.com/your_username/Dynamic_Feature_Fusion_Blockchain_Fraud.git
cd Dynamic_Feature_Fusion_Blockchain_Fraud
```

### 2. Prepare Data
- Place raw blockchain transaction data in the appropriate directory.
- Run the preprocessing pipeline:
```bash
python prepare_data.py
```
- Preprocessed files will be saved in:
```
data/preprocessed/Dataset/
```
(e.g., `weighted_adjacency_matrix.pkl`, `shuffled_clean_docs`).

### 3. Configure Environment
- Edit `env_config.py` to set:
  - Random seeds  
  - Data paths  
  - Offline mode for transformers (if required)

---

### 4. Train ETH-GBERT
```bash
python train.py --ds Dataset --dim 16 --lr 8e-6 --l2 0.001
```
**Options:**
- `--load 1` → Resume training from checkpoint  
- `--validate_program` → Quick validation (1 epoch)

📂 Checkpoints are saved in:
```
./output/
```
Logs include loss curves and evaluation metrics.

---

### 5. Train Baselines
**Random Forest**:
```bash
python train_random_forest.py --ds Dataset
```

**SVM**:
```bash
python train_svm.py --ds Dataset
```

---

### 6. Evaluation
All training scripts output classification reports with **Weighted F1**, **Precision**, and **Recall** for both validation and test sets.
