import pickle
import random
import importlib

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

from tqdm import tqdm
import sys

random.seed(0)
np.random.seed(0)

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.rcParams["figure.figsize"] = (20, 10)

ds_name = sys.argv[1]

is_sparse = False

with open("data/{}.pkl".format(ds_name), "rb") as f:
    ds = pickle.load(f)

    features = ds["features"]
    adjs = ds["adjs"]

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_relations = len(adjs)

from utils import *

# Preprocess features and adjacency matrices
features = preprocess_features(features)
adjs_norm = [normalized_laplacian(adj + np.eye(nb_nodes) * 3.0, is_sparse) for adj in adjs]

# Create tensors
features = torch.FloatTensor(features[np.newaxis])

if not is_sparse:
    adjs_norm = [torch.FloatTensor(adj[np.newaxis]) for adj in adjs_norm]
else:
    adjs_norm = [sparse_mx_to_torch_sparse_tensor(adj) for adj in adjs_norm]

if torch.cuda.is_available():
    features = features.cuda()
    adjs_norm = [adj.cuda() for adj in adjs_norm]

# Get labels for infomax
lbl_1 = torch.ones(nb_nodes)
lbl_0 = torch.zeros(nb_nodes)
infomax_labels = torch.cat((lbl_1, lbl_0))

if torch.cuda.is_available():
    infomax_labels = infomax_labels.cuda()

from mymodel import MyModel

if "biogrid_4211" in ds_name:
    hid_units = 32
    drop_prob = 0.0
    is_attn = False
    common_gcn = True
    normalize_z = False
    lr = 0.001
    l2_coef = 1e-05
    n_epochs = 2000
    patience = 100
    
elif "biogrid_4503_bis" in ds_name:
    hid_units = 128
    drop_prob = 0.0
    is_attn = False
    common_gcn = True
    normalize_z = False
    lr = 0.001
    l2_coef = 0.0
    n_epochs = 2000
    patience = 20

elif "dblp_5124" in ds_name:
    hid_units = 64
    drop_prob = 0.1
    is_attn = True
    common_gcn = True
    normalize_z = False
    lr = 0.001
    l2_coef = 1e-05
    n_epochs = 2000
    patience = 20

elif "imdb_3000" in ds_name:
    hid_units = 64
    drop_prob = 0.0
    is_attn = False
    common_gcn = True
    normalize_z = False
    lr = 0.001
    l2_coef = 1e-05
    n_epochs = 2000
    patience = 20

elif "STRING-DB_4083" in ds_name:
    hid_units = 64
    drop_prob = 0.1
    is_attn = False
    common_gcn = True
    normalize_z = False
    lr = 0.001
    l2_coef = 1e-05
    n_epochs = 2000
    patience = 20

model = MyModel(ft_size, hid_units, nb_relations, drop_prob, is_attn, common_gcn, normalize_z)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    model = model.cuda()

best_loss = 1e9
best_epoch = 0
cnt_wait = 0

bce_loss = nn.BCEWithLogitsLoss()
loss_history = []

for epoch in tqdm(range(n_epochs)):
    model.train()
    optimizer.zero_grad()

    # Shuffle features
    idx = np.random.permutation(nb_nodes)
    fts_shuf = features[:, idx, :]
    if torch.cuda.is_available():
        fts_shuf = fts_shuf.cuda()

    logits = model(features, adjs_norm, fts_shuf, is_sparse)

    # Compute loss
    loss = bce_loss(logits.squeeze(), infomax_labels)
    loss_history.append(loss.item())

    if loss < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        cnt_wait = 0

        torch.save(model.state_dict(), "results/best_mymodel_{}.pkl".format(ds_name))

    else:
        cnt_wait += 1

    if cnt_wait == patience:
        break
    
    loss.backward()
    optimizer.step()

model.load_state_dict(torch.load("results/best_mymodel_{}.pkl".format(ds_name)))

model.eval()
with torch.no_grad():
    z = model.embed(features, adjs_norm, is_sparse).cpu()
    torch.save(z, "results/embs/{}.pkl".format(ds_name))
        
    link_prediction_array = ds["link_prediction_array"]

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

z_a = z[0, link_prediction_array[:, 0]]
z_b = z[0, link_prediction_array[:, 1]]
labels = link_prediction_array[:, 2]

scores = []
for i in range(len(z_a)):
    scores.append(torch.sigmoid(torch.dot(z_a[i], z_a[i])))

print(ds_name)
print("Link prediction ROC-AUC :", roc_auc_score(labels, scores))
print("Average precision score :", average_precision_score(labels, scores))
print("")
