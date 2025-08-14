import argparse
import gc
import os
import pickle as pkl
import random
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import networkx as nx
from scipy.sparse import csr_matrix
from env_config import env_config

random.seed(env_config.GLOBAL_SEED)
np.random.seed(env_config.GLOBAL_SEED)

"""
Configuration
"""
parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="Dataset")
parser.add_argument("--validate_program", action="store_true")
args = parser.parse_args()
args.ds = args.ds
data_dir = f"data/preprocessed/{args.ds}"

print("SVM Start at:", time.asctime())

"""
Prepare data set
"""
print("\n----- Prepare data set -----")

objects = []
names = [
    "labels",
    "train_y",
    "train_y_prob",
    "valid_y",
    "valid_y_prob",
    "test_y",
    "test_y_prob",
    "shuffled_clean_docs",
    "address_to_index",
]
for i in range(len(names)):
    datafile = "./" + data_dir + "/data_%s.%s" % (args.ds, names[i])
    with open(datafile, "rb") as f:
        objects.append(pkl.load(f, encoding="latin1"))
(
    lables_list,
    train_y,
    train_y_prob,
    valid_y,
    valid_y_prob,
    test_y,
    test_y_prob,
    shuffled_clean_docs,
    address_to_index,
) = tuple(objects)

label2idx = lables_list[0]
idx2label = lables_list[1]

y = np.hstack((train_y, valid_y, test_y))

examples = []
for i, ts in enumerate(shuffled_clean_docs):
    examples.append(ts.strip())  # Extract text descriptions

# Load adjacency matrix for graph features
def load_data(filename):
    with open(filename, 'rb') as file:
        return pkl.load(file)

weighted_adj_matrix = load_data('data/preprocessed/Dataset/weighted_adjacency_matrix.pkl')
weighted_adj_matrix = csr_matrix(weighted_adj_matrix)  # Convert to sparse matrix
G = nx.from_scipy_sparse_array(weighted_adj_matrix)  # Use sparse array for efficiency

gc.collect()

# Feature Engineering
def engineer_features(docs, graph, address_to_index):
    features = []
    # Text features: TF-IDF
    tfidf = TfidfVectorizer(max_features=500)
    text_features = tfidf.fit_transform(docs).toarray()
    
    # Graph features: Degree centrality per node (account)
    centrality = nx.degree_centrality(graph)
    graph_features = [centrality.get(i, 0) for i in range(len(docs))]  # Use index as node ID; adjust if needed
    
    # Transaction features: Basic aggregate (e.g., length as proxy; expand with n-gram sums, etc.)
    trans_features = [len(doc.split()) for doc in docs]  # Placeholder
    
    # Combine features
    combined = np.hstack((text_features, np.array(graph_features).reshape(-1, 1), np.array(trans_features).reshape(-1, 1)))
    return combined

X = engineer_features(examples, G, address_to_index)

# Split data
train_size = len(train_y)
valid_size = len(valid_y)
X_train = X[:train_size]
X_valid = X[train_size : train_size + valid_size]
X_test = X[train_size + valid_size :]

"""
Train SVM
"""
print("\n----- Running training -----")

svm = SVC(kernel='linear', probability=True, random_state=env_config.GLOBAL_SEED)  # Linear kernel for efficiency; change to 'rbf' for non-linear
svm.fit(X_train, train_y)

"""
Evaluate
"""
def evaluate(X, y_true, dataset_name):
    y_pred = svm.predict(X)
    f1 = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"{dataset_name} weighted F1: {100 * f1:.3f}, Precision: {100 * prec:.3f}, Recall: {100 * rec:.3f}")
    return f1

print("\n----- Evaluation -----")
evaluate(X_valid, valid_y, "Valid_set")
evaluate(X_test, test_y, "Test_set")