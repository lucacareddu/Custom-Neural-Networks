''' 
Anomaly detection on MVTec AD using a custom Transformer model with manual backpropagation.

Official repo: https://github.com/lucacareddu/Custom-Neural-Networks

MVTec AD Dataset: https://huggingface.co/datasets/TheoM55/mvtec_anomaly_detection
'''

import os
import glob

import random
import numpy as np

import torch
import torch.nn as nn

from PIL import Image

from huggingface_hub import snapshot_download


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

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------- Data --------

category = "bottle"
img_size = 256
patch_size = 16
grid_size = img_size // patch_size
n_tokens = grid_size ** 2
patch_dim = patch_size ** 2 * 3


def download_data(category=category):
    return snapshot_download("TheoM55/mvtec_anomaly_detection", repo_type="dataset", allow_patterns=[f"*/{category}/*"])


def paths_from_data(root, category=category):
    train_paths = sorted(glob.glob(f"{root}/images/train/{category}/good/*.png"))
    test_paths = sorted(glob.glob(f"{root}/images/test/{category}/*/*.png"))
    mask_paths = [p.replace("/images/", "/masks/").replace(".png", "_mask.png") for p in test_paths]

    return train_paths, test_paths, mask_paths


def images_from_paths(paths, mask=False):
    resample = Image.NEAREST if mask else Image.BILINEAR
    images = []

    for p in paths:
        if mask and not os.path.exists(p):
            images.append(torch.zeros((img_size, img_size, 1)))
            continue

        im = Image.open(p).convert("L" if mask else "RGB").resize((img_size, img_size), resample)
        images.append(torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).reshape(img_size, img_size, -1) / 255.0)

    return torch.stack(images).permute(0, 3, 1, 2)


def patches_from_images(X):
    X = X.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

    return X.permute(0, 2, 3, 1, 4, 5).reshape(len(X), n_tokens, -1)


def images_from_patches(X):
    X = X.reshape(len(X), grid_size, grid_size, -1, patch_size, patch_size)

    return X.permute(0, 3, 1, 4, 2, 5).reshape(len(X), -1, img_size, img_size)


def dataset_from_paths(train_paths, test_paths, mask_paths):
    X_train = patches_from_images(images_from_paths(train_paths))
    X_test = patches_from_images(images_from_paths(test_paths))

    M_test = images_from_paths(mask_paths, mask=True).squeeze(1) > 0
    Y_test = torch.tensor([int(os.path.basename(os.path.dirname(p)) != "good") for p in test_paths])

    X_train_mean, X_train_std = X_train.mean(), X_train.std()

    X_train = (X_train - X_train_mean) / (X_train_std + 1e-7)
    X_test = (X_test - X_train_mean) / (X_train_std + 1e-7)

    return X_train, X_test, Y_test, M_test


# -------- Models --------

def layer_norm(h, g, b, epsilon=1e-7):
    h_std = (h.var(dim=-1, unbiased=False, keepdim=True) + epsilon).sqrt()
    h_norm = (h - h.mean(dim=-1, keepdim=True)) / h_std

    return h_norm * g + b, h_norm, h_std


def layer_norm_backward(dL_out, g, h_norm, h_std):
    dL_norm = dL_out * g
    dL_h = (dL_norm - dL_norm.mean(dim=-1, keepdim=True) -
            h_norm * (dL_norm * h_norm).mean(dim=-1, keepdim=True)) / h_std

    return dL_h, (dL_out * h_norm).sum(1).mean(0), dL_out.sum(1).mean(0)


class CustomTransformer(nn.Module):
    def __init__(self, i_dim=patch_dim, h_dim=128, o_dim=patch_dim, n_blocks=6):
        super().__init__()

        self.i_dim = i_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.n_blocks = n_blocks

        self.W1 = nn.Parameter(torch.empty((i_dim, h_dim)))
        self.pos = nn.Parameter(torch.empty((n_tokens, h_dim)))
        self.blocks = nn.ModuleList([nn.ParameterDict({
            "W_q": nn.Parameter(torch.empty((h_dim, h_dim))),
            "W_k": nn.Parameter(torch.empty((h_dim, h_dim))),
            "W_v": nn.Parameter(torch.empty((h_dim, h_dim))),
            "W_t": nn.Parameter(torch.empty((h_dim, h_dim))),
            "g_ln": nn.Parameter(torch.ones(h_dim)),
            "b_ln": nn.Parameter(torch.zeros(h_dim)),
            }) for _ in range(n_blocks)])
        self.g_ln = nn.Parameter(torch.ones(h_dim))
        self.b_ln = nn.Parameter(torch.zeros(h_dim))
        self.W2 = nn.Parameter(torch.empty((h_dim, o_dim)))

        for n, p in self.named_parameters():
            if not n.endswith(("g_ln", "b_ln")):
                torch.nn.init.trunc_normal_(p, std=h_dim**-0.5)

    def forward(self, X, mask_ratio=0.0):
        epsilon = 1e-7

        keep = (torch.rand(X.shape[:2], device=X.device) >= mask_ratio).float()
        h_pre1 = X * keep.unsqueeze(-1)
        h = h_pre1 @ self.W1 + self.pos

        return_tensors = {"h_pre1": h_pre1.detach(), "keep": keep}

        for l, block in enumerate(self.blocks):
            h_pre2, h_norm, h_std = layer_norm(h, block["g_ln"], block["b_ln"], epsilon)

            Q, K, V = h_pre2 @ block["W_q"], h_pre2 @ block["W_k"], h_pre2 @ block["W_v"]
            S = torch.nn.functional.softmax(Q @ K.mT / self.h_dim**0.5, dim=-1)
            U = S @ V

            h = U @ block["W_t"] + h

            return_tensors.update({f"h_std{l}": h_std.detach(), f"h_norm{l}": h_norm.detach(),
                                   f"h_pre2{l}": h_pre2.detach(), f"Q{l}": Q.detach(), f"K{l}": K.detach(),
                                   f"V{l}": V.detach(), f"S{l}": S.detach(), f"U{l}": U.detach()})

        h_2, h_norm, h_std = layer_norm(h, self.g_ln, self.b_ln, epsilon)
        z = h_2 @ self.W2

        loss = 0.5 * (z - X).pow(2).mean() # MSE

        return_tensors.update({"h_norm": h_norm.detach(), "h_std": h_std.detach(),
                               "h_2": h_2.detach(), "z": z.detach()})

        return loss, return_tensors


def block_backward(p, t, l, dL_h, h_dim):
    dL_U = dL_h @ p[f"blocks.{l}.W_t"].T
    dL_V = t[f"S{l}"].mT @ dL_U
    dL_S = dL_U @ t[f"V{l}"].mT

    dL_QK = t[f"S{l}"] * (dL_S - (dL_S * t[f"S{l}"]).sum(dim=-1, keepdim=True)) / h_dim**0.5

    dL_Q, dL_K = dL_QK @ t[f"K{l}"], dL_QK.mT @ t[f"Q{l}"]

    dL_h_pre2 = (dL_Q @ p[f"blocks.{l}.W_q"].T +
                 dL_K @ p[f"blocks.{l}.W_k"].T +
                 dL_V @ p[f"blocks.{l}.W_v"].T)

    dL_h_ln, dL_g_ln, dL_b_ln = layer_norm_backward(dL_h_pre2, p[f"blocks.{l}.g_ln"], t[f"h_norm{l}"], t[f"h_std{l}"])

    gradients = {
        f"blocks.{l}.W_t": (t[f"U{l}"].mT @ dL_h).mean(0),
        f"blocks.{l}.W_v": (t[f"h_pre2{l}"].mT @ dL_V).mean(0),
        f"blocks.{l}.W_k": (t[f"h_pre2{l}"].mT @ dL_K).mean(0),
        f"blocks.{l}.W_q": (t[f"h_pre2{l}"].mT @ dL_Q).mean(0),
        f"blocks.{l}.g_ln": dL_g_ln,
        f"blocks.{l}.b_ln": dL_b_ln,
        }

    return dL_h + dL_h_ln, gradients


def backward(X, named_parameters, named_forward_tensors):
    p = dict(named_parameters)
    t = dict(named_forward_tensors)

    h_dim = t["h_2"].shape[-1]
    n_blocks = len([n for n in p if n.endswith("W_t")])

    # ---< Intermediate >---

    dL_z = (t["z"] - X) / (X.shape[1] * X.shape[2])
    dL_h, dL_g_ln, dL_b_ln = layer_norm_backward(dL_z @ p["W2"].T, p["g_ln"], t["h_norm"], t["h_std"])

    # ---< Parameters >---

    gradients = {"W2": (t["h_2"].mT @ dL_z).mean(0), "g_ln": dL_g_ln, "b_ln": dL_b_ln}

    for l in reversed(range(n_blocks)):
        dL_h, block_gradients = block_backward(p, t, l, dL_h, h_dim)
        gradients.update(block_gradients)

    gradients["pos"] = dL_h.mean(0)
    gradients["W1"] = (t["h_pre1"].mT @ dL_h).mean(0)

    return gradients


# -------- Evaluation --------

def roc_auc(scores, labels):
    order = scores.argsort()
    labels = labels[order].double()

    _, inverse, counts = scores[order].unique_consecutive(return_inverse=True, return_counts=True)
    ranks = torch.arange(1, len(labels)+1, dtype=torch.float64)
    ranks = (torch.zeros(len(counts), dtype=torch.float64).scatter_add_(0, inverse, ranks) / counts)[inverse]

    n_pos = labels.sum()

    return (((ranks * labels).sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(labels) - n_pos))).item()


def errors_from_patches(f, X, mask_ratio, draws=8, batch_size=16, stats=None):
    errors, counts = torch.zeros(X.shape[:2]), torch.zeros(X.shape[:2])

    for _ in range(draws):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size].to(device)
            _, tensors = f(X_batch, mask_ratio=mask_ratio)

            masked = (1 - tensors["keep"]).cpu()
            errors[i:i+batch_size] += (tensors["z"] - X_batch).pow(2).mean(-1).cpu() * masked
            counts[i:i+batch_size] += masked

    errors = errors / counts.clamp(min=1)

    return errors if stats is None else (errors - stats[0]) / (stats[1] + 1e-7)


def metrics_from_errors(errors, Y_test, M_test):
    maps = errors.reshape(-1, grid_size, grid_size).repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)

    image_auc = roc_auc(errors.max(dim=-1).values, Y_test.bool())
    pixel_auc = roc_auc(maps.flatten(), M_test.flatten())

    return image_auc, pixel_auc


# -------- Training loop --------

root = download_data()
train_paths, test_paths, mask_paths = paths_from_data(root)
X_train, X_test, Y_test, M_test = dataset_from_paths(train_paths, test_paths, mask_paths)

f = CustomTransformer().to(device)

BATCH_SIZE = 16
iters = len(X_train) // BATCH_SIZE

lr = 3e-4
epochs = 400
mask_ratio = 0.5
optim = torch.optim.AdamW(f.parameters(), lr=lr)

grad_check = True

losses = []

train_every = 10
val_every = 100


for epoch in range(epochs):
    train_indices = torch.randperm(len(X_train))

    for it in range(iters):
        idx = train_indices[it*BATCH_SIZE:(it+1)*BATCH_SIZE]
        X = X_train[idx].to(device)

        loss_train, tensors_train = f(X, mask_ratio=mask_ratio)
        losses.append(loss_train.item())

        # Autograd
        if grad_check and epoch == 0:
            loss_train.backward()

        with torch.no_grad():
            # Ours
            gradients = backward(X, f.named_parameters(), tensors_train)

            for n, p in f.named_parameters():

                # Gradients correctness assessment
                if grad_check and epoch == 0 and not torch.allclose(p.grad, gradients[n], rtol=1e-3, atol=1e-9):
                    print(epoch, n)

                p.grad = gradients[n]

        optim.step()
        optim.zero_grad()

    mloss_train = torch.tensor(losses).mean()
    losses = []

    # Evaluation on train set
    if (epoch+1) % train_every == 0:
        print(f"Epoch {epoch+1} -> loss: {mloss_train.item()}")

    # Evaluation on test set
    if (epoch+1) % val_every == 0:
        with torch.no_grad():
            train_errors = errors_from_patches(f, X_train, mask_ratio)
            errors = errors_from_patches(f, X_test, mask_ratio, stats=(train_errors.mean(0), train_errors.std(0)))
            image_auc, pixel_auc = metrics_from_errors(errors, Y_test, M_test)
        print(f"    Eval - Epoch {epoch+1} -> loss: {mloss_train.item()}, Image AUROC: {image_auc}, Pixel AUROC: {pixel_auc}")
