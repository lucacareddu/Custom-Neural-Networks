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

from torchmetrics.functional import auroc

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

h_dim = 128
n_blocks = 6

class CustomTransformer(nn.Module):
    def __init__(self, i_dim=patch_dim, h_dim=h_dim, o_dim=patch_dim, n_blocks=n_blocks):
        super().__init__()

        self.i_dim = i_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.n_blocks = n_blocks

        self.W_in = nn.Parameter(torch.empty((i_dim, h_dim)))
        self.pos = nn.Parameter(torch.empty((n_tokens, h_dim)))
        self.blocks = nn.ModuleList([nn.ParameterDict({
            "W_Q": nn.Parameter(torch.empty((h_dim, h_dim))),
            "W_K": nn.Parameter(torch.empty((h_dim, h_dim))),
            "W_V": nn.Parameter(torch.empty((h_dim, h_dim))),
            "W_O": nn.Parameter(torch.empty((h_dim, h_dim))),
            "gamma": nn.Parameter(torch.ones(h_dim)),
            "beta": nn.Parameter(torch.zeros(h_dim)),
            }) for _ in range(n_blocks)])
        self.gamma = nn.Parameter(torch.ones(h_dim))
        self.beta = nn.Parameter(torch.zeros(h_dim))
        self.W_out = nn.Parameter(torch.empty((h_dim, o_dim)))

        for n, p in self.named_parameters():
            if not n.endswith(("gamma", "beta")):
                torch.nn.init.trunc_normal_(p, std=h_dim**-0.5)

    def forward(self, X, mask_ratio=0.0):
        epsilon = 1e-7

        keep = (torch.rand(X.shape[:2], device=X.device) >= mask_ratio).float()
        X_tilde = X * keep.unsqueeze(-1)
        h = X_tilde @ self.W_in + self.pos

        return_tensors = {"X_tilde": X_tilde.detach(), "keep": keep}

        for l, block in enumerate(self.blocks):
            sigma = (h.var(dim=-1, unbiased=False, keepdim=True) + epsilon).sqrt()
            h_hat = (h - h.mean(dim=-1, keepdim=True)) / sigma
            a = h_hat * block["gamma"] + block["beta"]

            Q, K, V = a @ block["W_Q"], a @ block["W_K"], a @ block["W_V"]
            S = torch.nn.functional.softmax(Q @ K.mT / self.h_dim**0.5, dim=-1)
            h = S @ V + a @ block["W_O"] + h

            return_tensors.update({f"sigma{l}": sigma.detach(), f"h_hat{l}": h_hat.detach(),
                                   f"a{l}": a.detach(), f"Q{l}": Q.detach(), f"K{l}": K.detach(),
                                   f"V{l}": V.detach(), f"S{l}": S.detach()})

        sigma = (h.var(dim=-1, unbiased=False, keepdim=True) + epsilon).sqrt()
        h_hat = (h - h.mean(dim=-1, keepdim=True)) / sigma
        a = h_hat * self.gamma + self.beta

        X_hat = a @ self.W_out

        loss = 0.5 * (X_hat - X).pow(2).mean() # MSE

        return_tensors.update({f"h_hat{self.n_blocks}": h_hat.detach(), f"sigma{self.n_blocks}": sigma.detach(),
                               f"a{self.n_blocks}": a.detach(), "X_hat": X_hat.detach()})

        return loss, return_tensors


def block_backward(p, t, l, dL_h, h_dim):
    dL_W_O = t[f"a{l}"].mT @ dL_h

    dL_V = t[f"S{l}"].mT @ dL_h
    dL_S = dL_h @ t[f"V{l}"].mT

    dL_QKt = t[f"S{l}"] * (dL_S - (dL_S * t[f"S{l}"]).sum(dim=-1, keepdim=True)) / h_dim**0.5

    dL_K = dL_QKt.mT @ t[f"Q{l}"]
    dL_Q = dL_QKt @ t[f"K{l}"]

    dL_W_V = t[f"a{l}"].mT @ dL_V
    dL_W_K = t[f"a{l}"].mT @ dL_K
    dL_W_Q = t[f"a{l}"].mT @ dL_Q

    dL_a = (dL_Q @ p[f"blocks.{l}.W_Q"].T +
                dL_K @ p[f"blocks.{l}.W_K"].T +
                dL_V @ p[f"blocks.{l}.W_V"].T +
                dL_h @ p[f"blocks.{l}.W_O"].T)

    dL_beta = dL_a.sum(1)
    dL_gamma = (dL_a * t[f"h_hat{l}"]).sum(1)
    dL_h_hat = dL_a * p[f"blocks.{l}.gamma"]

    dL_h = dL_h + (dL_h_hat - dL_h_hat.mean(dim=-1, keepdim=True) -
                   t[f"h_hat{l}"] * (dL_h_hat * t[f"h_hat{l}"]).mean(dim=-1, keepdim=True)) / t[f"sigma{l}"]

    gradients = {f"blocks.{l}.W_O": dL_W_O.mean(0),
                 f"blocks.{l}.W_V": dL_W_V.mean(0),
                 f"blocks.{l}.W_K": dL_W_K.mean(0),
                 f"blocks.{l}.W_Q": dL_W_Q.mean(0),
                 f"blocks.{l}.beta": dL_beta.mean(0),
                 f"blocks.{l}.gamma": dL_gamma.mean(0)}

    return dL_h, gradients


def backward(X, named_parameters, named_forward_tensors, n_blocks=n_blocks, h_dim=h_dim):
    p = dict(named_parameters)
    t = dict(named_forward_tensors)

    dL_X_hat = (t["X_hat"] - X) / (X.shape[1] * X.shape[2])
    dL_W_out = t[f"a{n_blocks}"].mT @ dL_X_hat
    dL_a = dL_X_hat @ p["W_out"].T

    dL_beta = dL_a.sum(1)
    dL_gamma = (dL_a * t[f"h_hat{n_blocks}"]).sum(1)
    dL_h_hat = dL_a * p["gamma"]
    dL_h = (dL_h_hat - dL_h_hat.mean(dim=-1, keepdim=True) -
            t[f"h_hat{n_blocks}"] * (dL_h_hat * t[f"h_hat{n_blocks}"]).mean(dim=-1, keepdim=True)) / t[f"sigma{n_blocks}"]

    gradients = {"W_out": dL_W_out.mean(0), "beta": dL_beta.mean(0), "gamma": dL_gamma.mean(0)}

    for l in reversed(range(n_blocks)):
        dL_h, block_gradients = block_backward(p, t, l, dL_h, h_dim)
        gradients.update(block_gradients)

    dL_pos = dL_h
    dL_W_in = t["X_tilde"].mT @ dL_h

    gradients.update({"pos": dL_pos.mean(0), "W_in": dL_W_in.mean(0)})

    return gradients


# -------- Metrics --------

@torch.no_grad()
def errors_from_patches(f, X, mask_ratio, draws=8, batch_size=16):
    errors, counts = torch.zeros(X.shape[:2]), torch.zeros(X.shape[:2])

    for _ in range(draws if mask_ratio else 1):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size].to(device)
            _, tensors = f(X_batch, mask_ratio=mask_ratio)

            masked = (1 - tensors["keep"]).cpu() if mask_ratio else torch.ones(X_batch.shape[:2])
            errors[i:i+batch_size] += (tensors["X_hat"] - X_batch).pow(2).mean(-1).cpu() * masked
            counts[i:i+batch_size] += masked

    return errors / counts.clamp(min=1)


def fused_errors(f, X):
    errors = [errors_from_patches(f, X, r) for r in eval_mask_ratios]
    # median so every ratio map weighs equally in the average
    errors = sum(e / e.median(-1, keepdim=True).values.clamp(min=1e-6) for e in errors) / len(errors)
    # median so a defect does not cancel its own signal
    errors = errors / errors.median(-1, keepdim=True).values.clamp(min=1e-6)

    # 3x3 box blur with zero padding biasing the score toward the center
    blurred_errors = nn.functional.avg_pool2d(errors.reshape(-1, 1, grid_size, grid_size), 3, stride=1, padding=1).reshape(len(errors), -1)
    
    return blurred_errors


def largest_blob(errors, tau):
    '''Iterative connected-component labeling (CCL) using label propagation.'''

    mask = (errors > tau).float().reshape(-1, grid_size, grid_size)
    labels = (torch.arange(n_tokens, dtype=torch.float32).reshape(1, grid_size, grid_size) + 1) * mask

    for _ in range(2 * grid_size):
        propagated = nn.functional.max_pool2d(labels.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) * mask
        if torch.equal(propagated, labels):
            break
        labels = propagated

    labels = labels.flatten(1)
    largest_sizes = torch.stack([l[l > 0].unique(return_counts=True)[1].max().float() if (l > 0).any() else torch.zeros(()) for l in labels])

    return largest_sizes


def metrics_from_errors(train_errors, errors, Y_test, M_test):
    maps = nn.functional.interpolate(errors.reshape(-1, 1, grid_size, grid_size), size=img_size, mode="bilinear", align_corners=False).squeeze(1)

    threshold = torch.quantile(train_errors.flatten(), 0.99) # max has less support
    largest_sizes = largest_blob(errors, threshold)

    image_auc = auroc(largest_sizes, Y_test, task="binary") # auroc by largest blob size
    pixel_auc = auroc(maps.flatten(), M_test.flatten().int(), task="binary") # auroc by patch loss

    return image_auc.item(), pixel_auc.item()


# -------- Training loop --------

root = download_data()
train_paths, test_paths, mask_paths = paths_from_data(root)
X_train, X_test, Y_test, M_test = dataset_from_paths(train_paths, test_paths, mask_paths)

f = CustomTransformer().to(device)

BATCH_SIZE = 16
iters = len(X_train) // BATCH_SIZE

lr = 1e-3
epochs = 400
mask_ratio = 0.5
eval_mask_ratios = [0.0, 0.25, 0.5, 0.75]
optim = torch.optim.AdamW(f.parameters(), lr=lr)

grad_check = True

losses = []

train_every = 10
val_every = 100

os.makedirs("weights", exist_ok=True)


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
            train_errors = fused_errors(f, X_train)
            errors = fused_errors(f, X_test)
            image_auc, pixel_auc = metrics_from_errors(train_errors, errors, Y_test, M_test)
        print(f"    Eval - Epoch {epoch+1} -> loss: {mloss_train.item()}, Image AUROC: {image_auc}, Pixel AUROC: {pixel_auc}")

torch.save(f.state_dict(), f"weights/{category}_{mask_ratio}.pt")
