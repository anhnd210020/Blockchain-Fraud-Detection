# Blockchain-Fraud-Detection
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
