# RUL prediction (machine failure dataset) using a custom Transformer model
# with manual backpropagation

import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

seed = 42 # any number 
# set_deterministic(seed=seed)


# -------- Data --------

data = pd.read_csv("archive/machine failure.csv")

data = data.sort_values('UDI')

cols = ['Type', 'Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 
        'Torque [Nm]', 'Tool wear [min]', 'Machine failure']

data = data[cols]


#  FEATURE ENGINEERING #

# print([(col, data[col].skew()) for col in data.columns])

data['Type'] = data['Type'].replace(['M', 'L', 'H'], [1,2,3])
data['delta_temperature'] = data['Process temperature [K]'] - data['Air temperature [K]']
data['power'] = data['Torque [Nm]'] * data['Rotational speed [rpm]']

session_id = (data['Tool wear [min]'] < data['Tool wear [min]'].shift(fill_value=0)).cumsum()
sessions = data.groupby(session_id)
fail_id = sessions['Machine failure'].shift(fill_value=0).cumsum()
rtf = data.groupby(by=[session_id, fail_id])

# delta_wear (Tool Wear) column
rtf = [r.assign(delta_wear = r['Tool wear [min]'] - r['Tool wear [min]'].shift(fill_value=r.reset_index().loc[0,'Tool wear [min]'])) 
        for _, r in rtf if r['Machine failure'].sum()]

# RUL (cycles) column
rtf = [r.assign(RUL = np.arange(len(r))[::-1]) 
        for r in rtf]


#  DATASET CONSTRUCTION #

# rtf_cycles = [x.shape[0] for x in rtf]
# print(len(rtf_cycles), np.mean(rtf_cycles), np.std(rtf_cycles), np.min(rtf_cycles), np.median(rtf_cycles), np.max(rtf_cycles))

new_cols = ['Type', 'Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 
        'Torque [Nm]', 'delta_temperature', 'power', 'delta_wear']

min_cycles = 5
target_seq_len = 10

X = []
y = []

for r in rtf:
    r.reset_index(drop=True, inplace=True)
    r.drop(index=len(r)-1, inplace=True)

    n_splits = len(r) // target_seq_len + int(len(r) % target_seq_len >= min_cycles)

    start_indices = [target_seq_len*i for i in range(n_splits)]

    for start_idx in start_indices:
        end_idx = min(len(r)-1, start_idx + target_seq_len - 1)

        sample = r.loc[start_idx:end_idx, new_cols]
        sample["id"] = list(range(1,len(sample)+1)) # sequence id column
        sample = sample.to_numpy(dtype=np.float32)

        # Pad sequences to target length
        if len(sample) < target_seq_len:
            sample = np.pad(sample, ((0, target_seq_len - len(sample)),(0,0)))
            
        label = r.loc[end_idx, 'RUL'].astype(np.int64)

        X.append(sample)
        y.append(label)

# print(len(y), np.mean(y), np.median(y))

shuffled_indices = np.random.permutation(len(X))
X = torch.tensor(np.asarray(X))[shuffled_indices]
y = torch.tensor(np.asarray(y))[shuffled_indices]

train_indices = np.arange(int(len(X)*0.8))
test_indices = np.arange(int(len(X)*0.8), len(X))

X_train, Y_train = X[train_indices], y[train_indices]
X_test, Y_test = X[test_indices], y[test_indices]

# Padding Masks
X_train_mask = ~torch.all(X_train == 0, dim=-1)
X_test_mask = ~torch.all(X_test == 0, dim=-1)

X_train_mean, X_train_std = X_train[X_train_mask].mean(dim=0), X_train[X_train_mask].std(dim=0)

# Standardization
X_train[X_train_mask] = (X_train[X_train_mask] - X_train_mean) / (X_train_std + 1e-7)
X_test[X_test_mask] = (X_test[X_test_mask] - X_train_mean) / (X_train_std + 1e-7)


# print(X_train[0])


# -------- Models --------

class CustomTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        i_dim = 9
        h_dim = 18
        o_dim = 1

        self.i_dim = i_dim
        self.h_dim = h_dim
        self.o_dim = o_dim

        self.W1 = nn.Parameter(torch.empty((i_dim, h_dim)))
        self.W_q, self.W_k, self.W_v = nn.Parameter(torch.empty((h_dim, h_dim))), nn.Parameter(torch.empty((h_dim, h_dim))), nn.Parameter(torch.empty((h_dim, h_dim)))
        self.W_t = nn.Parameter(torch.empty((h_dim, h_dim)))
        self.W2 = nn.Parameter(torch.empty((h_dim, o_dim)))
        
        torch.nn.init.trunc_normal_(self.W1)
        torch.nn.init.trunc_normal_(self.W_q)
        torch.nn.init.trunc_normal_(self.W_k)
        torch.nn.init.trunc_normal_(self.W_v)
        torch.nn.init.trunc_normal_(self.W_t)
        torch.nn.init.trunc_normal_(self.W2)

    def forward(self, X, y):
        mask = torch.all(X == 0, dim=-1)
        attn_mask = mask.unsqueeze(-1) | mask.unsqueeze(1)
        point_mask = (~mask).unsqueeze(-1)

        h_pre1 = X
        h_1 = h_pre1 @ self.W1
        h_pre2 = h_1
        Q, K, V, T = h_pre2 @ self.W_q, h_pre2 @ self.W_k, h_pre2 @ self.W_v, h_pre2 @ self.W_t

        scores = Q @ K.mT / self.h_dim**0.5
        scores = scores.masked_fill(attn_mask, float('-inf'))

        S = torch.nn.functional.softmax(scores, dim=-1)
        S = S.masked_fill(attn_mask, 0)
        
        V = V * point_mask.float()
        h_2 = S @ V + h_pre2

        h_2_mean = (h_2 * point_mask.float()).sum(dim=1) / point_mask.sum(dim=1)
        z_pre = h_2_mean @ self.W2

        z = nn.functional.relu(z_pre)

        loss = 0.5 * (y - z.squeeze()).pow(2).mean() # MSE

        return_tensors = {
            "h_pre1": h_pre1.detach(),
            "h_1": h_1.detach(),
            "h_pre2": h_pre2.detach(),
            "Q": Q.detach(),
            "K": K.detach(),
            "V": V.detach(),
            "S": S.detach(),
            "T": T.detach(),
            "h_2": h_2.detach(),
            "z": z.detach(),
        }

        return loss, return_tensors


def backward(X, y, named_parameters, named_forward_tensors):
    p = dict(named_parameters)
    t = dict(named_forward_tensors)

    h_dim = t["K"].shape[-1]

    y = y.unsqueeze(1)

    # ---< Intermediate >---

    dL_z = (t["z"] - y).unsqueeze(1)
    dL_h_2 = dL_z @ p["W2"].T
    dL_T = dL_h_2
    dL_V = t["S"].mT[:,:,-1:] @ dL_h_2
    dL_S = dL_h_2 @ t["V"].mT

    # S * (g - g @ S.mT) / K_dim
    dL_QK = t["S"] * (dL_S - (dL_S * t["S"]).sum(dim=-1, keepdim=True)) / h_dim**0.5  # (B, seq_len, seq_len)

    dL_K = dL_QK.mT[:,:,-1:] @ t["Q"][:,-1:]
    dL_Q = dL_QK[:,-1:] @ t["K"]

    # ---< Parameters >---

    dL_W2 = t["h_2"].mT[:,:,-1:] @ dL_z
    dL_W_t = t["h_pre2"].mT[:,:,-1:] @ dL_T
    dL_W_v = t["h_pre2"].mT @ dL_V
    dL_W_k = t["h_pre2"].mT @ dL_K
    dL_W_q = t["h_pre2"].mT[:,:,-1:] @ dL_Q
    dL_cls_tok = (dL_Q @ p["W_q"].T +
                  dL_K[:,-1:] @ p["W_k"].T +
                  dL_V[:,-1:] @ p["W_v"].T +
                  dL_T @ p["W_t"].T)
    dL_W1 = (t["h_pre1"].mT @ dL_K[:,:-1] @ p["W_k"].T +
             t["h_pre1"].mT @ dL_V[:,:-1] @ p["W_v"].T)

    gradients = {
        "W2": dL_W2.mean(0),
        "W_t": dL_W_t.mean(0),
        "W_v": dL_W_v.mean(0),
        "W_k": dL_W_k.mean(0),
        "W_q": dL_W_q.mean(0),
        "cls_tok": dL_cls_tok.mean(0).squeeze(),
        "W1": dL_W1.mean(0),
        }

    return gradients


f = CustomTransformer()


# -------- Training loop --------

BATCH_SIZE = 32
iters = len(X_train) // BATCH_SIZE

train_indices = torch.randperm(len(X_train))

lr = 1e-2
epochs = 100
optim = torch.optim.AdamW(f.parameters(), lr=lr) # used only for zero_grad()

losses, scores = [], []

# parameters_names = list(n for n,_ in f.named_parameters())

# alpha_mom = 0.8
# momentum = dict(zip(parameters_names, [0]*len(parameters_names)))

train_every = 1
val_every = 10


for epoch in range(epochs):
    for it in range(iters):
        idx = train_indices[it*BATCH_SIZE:(it+1)*BATCH_SIZE]
        X = X_train[idx]
        Y = Y_train[idx]

        loss_train, tensors_train = f(X, Y)

        losses.append(loss_train.item())
        scores.append(math.sqrt(loss_train.item()))

        # Autograd
        loss_train.backward()

        # with torch.no_grad():
        #     # Ours
        #     gradients = backward(X, Y, f.named_parameters(), tensors_train)
            
        #     for n, p in f.named_parameters():

        #         # Gradients correctness assessment
        #         if not torch.allclose(p.grad, gradients[n], atol=1e-4):
        #             print(epoch, n)
                
        #         # Gradient Descent
        #         p -= lr * gradients[n]
        # #         momentum[n] = alpha_mom * momentum[n] + (1-alpha_mom) * gradients[n]
        # #         momentum_n_corrected = momentum[n] / (1 - alpha_mom ** ((epoch*iters)+it+1))
        # #         p -= lr * momentum_n_corrected

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(f.parameters(), 1)

        optim.step()
        optim.zero_grad()
    
    mloss_train, mrmse_train = torch.tensor(losses).mean(), torch.tensor(scores).mean()
    losses, scores = [], []

    # Evaluation on train set
    if (epoch+1) % train_every == 0:
        print(f"Epoch {epoch+1} -> loss: {mloss_train.item()}, RMSE: {mrmse_train.item()}")

    # Evaluation on test set
    if (epoch+1) % val_every == 0:
        loss_val, tensors_val = f(X_test, Y_test)
        rmse_val = math.sqrt(loss_val.item())

        print(f"    Eval (Train/Val) - Epoch {epoch+1} -> loss: {mloss_train.item()}/{loss_val.item()}, RMSE: {mrmse_train.item()}/{rmse_val}")
        
        # plot validation set distribution overlap
        sns.histplot(Y_test, label="true")
        sns.histplot(tensors_val["z"].squeeze(), label="pred")
        plt.legend()
        plt.show()
