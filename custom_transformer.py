import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function

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
set_deterministic(seed=seed)


# -------- Data --------

time_steps = 100
num_features = 8
n_train = 1024
n_val = 100

def generate_dataset(n_samples):
    X = torch.zeros(n_samples, time_steps, num_features)
    y = torch.zeros(n_samples, dtype=torch.long)

    t = torch.linspace(0, 1, time_steps)

    for i in range(n_samples):
        # Latent regime
        regime = torch.randint(0, 2, (1,)).item()

        # Base signals
        trend = torch.randn(1) * t
        seasonality = torch.sin(2 * math.pi * (t * (2 + torch.rand(1))))

        # Autoregressive components
        ar = torch.zeros(time_steps)
        ar[0] = torch.randn(1)
        for k in range(1, time_steps):
            ar[k] = 0.8 * ar[k - 1] + 0.2 * torch.randn(1)

        # Feature construction
        X[i, :, 0] = ar + 0.1 * torch.randn(time_steps)
        X[i, :, 1] = seasonality + 0.1 * torch.randn(time_steps)
        X[i, :, 2] = trend + 0.1 * torch.randn(time_steps)
        X[i, :, 3] = ar * seasonality
        X[i, :, 4] = torch.randn(time_steps)

        # Event-based feature (rare spike)
        event_time = torch.randint(10, time_steps - 10, (1,))
        X[i, event_time:event_time+3, 5] += torch.randn(3) * 3

        # Regime-dependent features
        if regime == 1:
            X[i, :, 6] = torch.cos(4 * math.pi * t) + 0.2 * ar
            X[i, :, 7] = 0.5 * trend + 0.3 * seasonality
        else:
            X[i, :, 6] = torch.sin(6 * math.pi * t) - 0.2 * ar
            X[i, :, 7] = -0.3 * trend

        # Label logic (delayed + interaction-based)
        signal = (
            X[i, 40:, 0].mean() +
            0.5 * X[i, :, 3].std() +
            0.8 * X[i, :, 5].max()
        )

        y[i] = 1 if signal > 1.0 else 0

    return X, y

# Generate datasets
X_train, Y_train = generate_dataset(n_train)
X_val, Y_val = generate_dataset(n_val)

# print(torch.bincount(y_train))


# import matplotlib.pyplot as plt

# features = X_train.shape[2]

# # Combine all data (train + val) or just use X_train
# X = X_train
# y = y_train

# # Indices for each class
# class0_idx = (y == 0).nonzero(as_tuple=True)[0]
# class1_idx = (y == 1).nonzero(as_tuple=True)[0]

# # Create figure
# fig, axes = plt.subplots(features, 2, figsize=(12, 2*features), sharex=True)

# for f in range(features):
#     # Class 0
#     for idx in class0_idx[:50]:  # plot first 50 samples lightly
#         axes[f, 0].plot(X[idx, :, f], color='blue', alpha=0.2)
#     axes[f, 0].plot(X[class0_idx, :, f].mean(0), color='blue', linewidth=2)
#     axes[f, 0].set_title(f"Feature {f} | Class 0")
    
#     # Class 1
#     for idx in class1_idx[:50]:
#         axes[f, 1].plot(X[idx, :, f], color='red', alpha=0.2)
#     axes[f, 1].plot(X[class1_idx, :, f].mean(0), color='red', linewidth=2)
#     axes[f, 1].set_title(f"Feature {f} | Class 1")

# plt.tight_layout()
# plt.show()


# -------- Models --------

# # Inherit from Function
# class CustomFunction(Function):

#     # Note that forward, setup_context, and backward are @staticmethods
#     @staticmethod
#     def forward(input, labels, *parameters):
#         # output = input.mm(weight.t())
#         # if bias is not None:
#         #     output += bias.unsqueeze(0).expand_as(output)
#         # return output
#         pass

#     @staticmethod
#     # inputs is a Tuple of all of the inputs passed to forward.
#     # output is the output of the forward().
#     def setup_context(ctx, inputs, output):
#         ctx.params = {}
#         input, weight, bias = inputs
#         ctx.save_for_backward(input, weight, bias)

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         # This is a pattern that is very convenient - at the top of backward
#         # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#         # None. Thanks to the fact that additional trailing Nones are
#         # ignored, the return statement is simple even when the function has
#         # optional inputs.
#         input, weight, bias = ctx.saved_tensors
#         grad_input = grad_weight = grad_bias = None

#         # These needs_input_grad checks are optional and there only to
#         # improve efficiency. If you want to make your code simpler, you can
#         # skip them. Returning gradients for inputs that don't require it is
#         # not an error.
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.mm(weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.t().mm(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)

#         return grad_input, grad_weight, grad_bias


class CustomTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        i_dim = 8
        h_dim = 32
        o_dim = 2

        self.i_dim = i_dim
        self.h_dim = h_dim
        self.o_dim = o_dim

        self.W1 = nn.Parameter(torch.empty((i_dim, h_dim)))
        self.cls_tok = nn.Parameter(torch.randn(h_dim))
        self.W_q, self.W_k, self.W_v = nn.Parameter(torch.empty((h_dim, h_dim))), nn.Parameter(torch.empty((h_dim, h_dim))), nn.Parameter(torch.empty((h_dim, h_dim)))
        self.W_t = nn.Parameter(torch.empty((h_dim, h_dim)))
        self.W2 = nn.Parameter(torch.empty((h_dim, o_dim)))
        
        torch.nn.init.trunc_normal_(self.W1)
        torch.nn.init.trunc_normal_(self.cls_tok)
        torch.nn.init.trunc_normal_(self.W_q)
        torch.nn.init.trunc_normal_(self.W_k)
        torch.nn.init.trunc_normal_(self.W_v)
        torch.nn.init.trunc_normal_(self.W_t)
        torch.nn.init.trunc_normal_(self.W2)

        # self.fn = CustomFunction.apply

    def forward(self, X, y):
        # return self.fn(x, y, *self.named_parameters())

        B = X.shape[0]

        epsilon = 1e-7

        h_pre1 = (X - X.mean()) / (X.std() + epsilon)
        h_1 = h_pre1 @ self.W1
        h_pre2 = torch.concat([h_1, self.cls_tok.unsqueeze(0).unsqueeze(0).expand(B,1,-1)], dim=1)
        Q, K, V = h_pre2 @ self.W_q, h_pre2 @ self.W_k, h_pre2 @ self.W_v
        S = torch.nn.functional.softmax(Q @ K.mT / self.h_dim**0.5, dim=-1)
        T = h_pre2 @ self.W_t
        h_2 = S @ V + T
        z = h_2[:,-1] @ self.W2

        log_s = torch.nn.functional.log_softmax(z, dim=-1) # log-sum-exp trick

        # if self.K > 1 and not y.dim() > 1:
        #     y = torch.nn.functional.one_hot(y, 2)

        # loss = - torch.mean(torch.sum(y * log_s, dim=-1))

        loss = - log_s.gather(1, y.unsqueeze(1)).squeeze(1).mean() # computes the same as above but more efficiently

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
            "s": log_s.detach().exp()
        }

        return loss, return_tensors


def backward(X, y, named_parameters, named_forward_tensors):
    p = dict(named_parameters)
    t = dict(named_forward_tensors)

    h_dim = t["K"].shape[-1]

    y = torch.nn.functional.one_hot(y, 2)

    # ---< Intermediate >---

    dL_z = (t["s"] - y).unsqueeze(1)
    dL_h_2 = dL_z @ p["W2"].T
    dL_V = t["S"][:,-1].unsqueeze(2) @ dL_h_2
    dL_S = dL_h_2 @ t["V"].mT

    # S * (g - g @ S.mT) / K_dim
    dL_QK = t["S"] * (dL_S - (dL_S * t["S"]).sum(dim=-1, keepdim=True)) / h_dim**0.5  # (B, seq_len, seq_len)

    dL_K = t["Q"][:,-1].unsqueeze(2) @ dL_QK[:,-1:]
    dL_Q = dL_QK[:,-1:] @ t["K"]

    # ---< Parameters >---

    dL_W2 = t["h_2"][:,-1].unsqueeze(2) @ dL_z
    dL_W_t = t["h_pre2"][:,-1].unsqueeze(2) @ dL_h_2
    dL_W_v = t["h_pre2"].mT @ dL_V
    dL_W_k = t["h_pre2"].mT @ dL_K.mT
    dL_W_q = t["h_pre2"][:,-1].unsqueeze(2) @ dL_Q
    dL_cls_tok = (dL_Q @ p["W_q"].T +
               dL_K.mT[:,-1:] @ p["W_k"].T +
               dL_V[:,-1:] @ p["W_v"].T +
               dL_h_2 @ p["W_t"].T)
    dL_W1 = (t["h_pre1"].mT @ dL_K.mT[:,:-1] @ p["W_k"].T +
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

lr = 1e-3
epochs = 100

optim = torch.optim.SGD(f.parameters(), lr=lr) # used only for zero_grad()

train_every = 1
eval_every = 100

for epoch in range(epochs):
    loss_train, tensors_train = f(X_train, Y_train)

    if (epoch+1) % train_every == 0:
        pred_train = tensors_train["s"]
        mAcc_train = (torch.argmax(pred_train, dim=-1) == Y_train).float().mean()
        print(f"Epoch {epoch+1} -> loss: {loss_train.item()}, mAcc: {mAcc_train.item()}")

    # Autograd
    loss_train.backward()

    with torch.no_grad():
        # Ours
        gradients = backward(X_train, Y_train, f.named_parameters(), tensors_train)
        
        for n, p in f.named_parameters():

            # Gradients correctness assessment
            if not torch.allclose(p.grad, gradients[n], atol=1e-6):
                print(epoch, n)
            
            # Gradient Descent
            p -= lr * gradients[n]

        # Evaluation on test set
        if (epoch+1) % eval_every == 0:
            loss_val, tensors_val = f(X_val, Y_val)
            pred_val = tensors_val["s"]
            mAcc_val = (torch.argmax(pred_val, dim=-1) == Y_val).float().mean()
            print(f"    Eval (Train/Val) - Epoch {epoch+1} -> loss: {loss_train.item()}/{loss_val.item()}, mAcc: {mAcc_train.item()}/{mAcc_val.item()}")

    # optim.step()
    optim.zero_grad()
