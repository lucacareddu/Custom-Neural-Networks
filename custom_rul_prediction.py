''' 
RUL prediction using a custom Transformer model with manual backpropagation.

Official repo: https://github.com/lucacareddu/Custom-Neural-Networks

Machine Failure Dataset: https://www.kaggle.com/datasets/shashanknecrothapa/machine-failure-predictions/code
'''

import os
import shutil
import datetime

import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from mlflow.models.signature import infer_signature

os.environ['PREFECT_API_URL'] = ''
os.environ['PREFECT_API_KEY'] = ''
from prefect import task, flow

import logging

import ast


# -------- Timestamp --------

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# -------- Reproducibility --------

reproducible_run = True
seed = 42

@task
def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# -------- MLFlow Setup --------

@task
def setup_mlflow():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # os.environ["AWS_ACCESS_KEY_ID"] = ""
    # os.environ["AWS_SECRET_ACCESS_KEY"] = ""
    # os.environ["AWS_DEFAULT_REGION"] = "eu-north-1"

    logger = logging.getLogger("mlflow")

    # Set log level to *
    logger.setLevel(logging.ERROR)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RUL Estimation Demo")

# -------- Data --------

min_cycles = 5
target_seq_len = 10

@task()
def load_data(path=None, stream=None, inference=False):
    assert (path and not inference) or (stream and inference)
    
    data = pd.read_csv(path) if not inference else pd.read_json(stream, orient="records")

    # data[['Type', 'Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 
    #     'Torque [Nm]', 'Tool wear [min]']].head(10).to_json("stream_test", orient="records", indent=2)

    data = data.sort_values('UDI') if not inference else data

    cols = ['Type', 'Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 
            'Torque [Nm]', 'Tool wear [min]']
    
    cols = cols + ['Machine failure'] if not inference else cols

    data = data[cols]

    return data

@task()
#  FEATURE ENGINEERING
def runs_from_data(data, inference=False):

    # print([(col, data[col].skew()) for col in data.columns])

    data['Type'] = data['Type'].replace(['M', 'L', 'H'], [1,2,3])
    data['delta_temperature'] = data['Process temperature [K]'] - data['Air temperature [K]']
    data['power'] = data['Torque [Nm]'] * data['Rotational speed [rpm]']

    session_id = (data['Tool wear [min]'] < data['Tool wear [min]'].shift(fill_value=0)).cumsum()
    sessions = data.groupby(session_id)
    
    data["delta_wear"] = sessions['Tool wear [min]'].transform(lambda x: x - x.shift(fill_value=x.iloc[0]))
    
    if not inference:
        fail_id = sessions['Machine failure'].shift(fill_value=0).cumsum()
        runs = data.groupby(by=[session_id, fail_id])

        # RUL (cycles) column
        runs = [r.assign(RUL = np.arange(len(r))[::-1]) for _,r in runs if r['Machine failure'].sum()]
    else:
       runs = [r for _,r in data.groupby(session_id)]

    return runs

@task
#  SEQUENCES CRAFTING
def sequences_from_runs(runs, min_cycles = 5, target_seq_len = 10, inference=False):
    min_cycles = min_cycles if not inference else 1

    # runs_cycles = [x.shape[0] for x in runs]
    # print(len(runs_cycles), np.mean(runs_cycles), np.std(runs_cycles), np.min(runs_cycles), np.median(runs_cycles), np.max(runs_cycles))

    new_cols = ['Type', 'Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 
            'Torque [Nm]', 'delta_temperature', 'power', 'delta_wear']

    X = []
    y = []

    for r in runs:
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
                
            X.append(sample)

            if not inference:
                label = r.loc[end_idx, 'RUL'].astype(np.int64)
                y.append(label)

    # print(len(y), np.mean(y), np.median(y))

    return X, y

@task(persist_result=False)
#  Training / Validation
def dataset_from_sequences(X,y):
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

    str_mean = str(X_train_mean.numpy().tolist())
    str_std = str(X_train_std.numpy().tolist())

    mlflow.log_params({"X_train_mean": str_mean, "X_train_std": str_std})

    # Standardization
    X_train[X_train_mask] = (X_train[X_train_mask] - X_train_mean) / (X_train_std + 1e-7)
    X_test[X_test_mask] = (X_test[X_test_mask] - X_train_mean) / (X_train_std + 1e-7)

    # print(X_train[0])

    return X_train, Y_train, X_test, Y_test

@task
#  Inference
def tensors_from_sequences(X, y, normalize=True, mlflow_run_id=None):
    assert not normalize or (normalize and mlflow_run_id)

    X = torch.tensor(np.asarray(X))
    y = torch.tensor(np.asarray(y)) if y else y

    if normalize:
        # Padding Mask
        X_mask = ~torch.all(X == 0, dim=-1)

        mean = mlflow.get_run(mlflow_run_id).data.params.get("X_train_mean")
        std = mlflow.get_run(mlflow_run_id).data.params.get("X_train_std")

        assert mean and std

        mean = torch.from_numpy(np.array(ast.literal_eval(mean), dtype=np.float32))
        std = torch.from_numpy(np.array(ast.literal_eval(std), dtype=np.float32))

        # Standardization
        X[X_mask] = (X[X_mask] - mean) / (std + 1e-7)
    
    return X, y

@flow
def data_pipeline(path="archive/machine failure.csv", stream=None, mlflow_run_id=None, inference=False):
    data = load_data(path=path, stream=stream, inference=inference)
    runs = runs_from_data(data, inference=inference)
    X, y = sequences_from_runs(runs, inference=inference)

    if not inference:
        X_train, Y_train, X_test, Y_test = dataset_from_sequences(X, y)
        return X_train, Y_train, X_test, Y_test
    else:
        X, y = tensors_from_sequences(X, y, mlflow_run_id=mlflow_run_id)
        return X, y


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


# -------- Train-Infer Functions --------

@flow(flow_run_name=timestamp, retries=1)
def infer(stream="stream_test", model_name="Adam_W", alias="production"):
    setup_mlflow()

    model_uri = f"models:/{model_name}@{alias}"

    model = mlflow.pyfunc.load_model(model_uri)
    run_id = model.metadata.run_id

    X_test, _ = data_pipeline(stream=stream, inference=True, mlflow_run_id=run_id)

    model = model.get_raw_model().to("cpu")
    model.eval()    

    with torch.no_grad():
        _, forward_tensors = model(X_test, torch.rand(X_test.shape[0]))
        pred = forward_tensors["z"].squeeze().floor().int().tolist()
    
    return pred


@flow(flow_run_name=timestamp, retries=1)
def train():
    if reproducible_run:
        set_deterministic(seed=seed)

    setup_mlflow()
    
    # Log CPU, GPU, Disk and Network metrics
    mlflow.enable_system_metrics_logging()

    with mlflow.start_run(run_name=timestamp):

        # -------- Initialization --------

        mlflow.log_params({"reproducible_run": reproducible_run, "seed": seed})

        X_train, Y_train, X_test, Y_test = data_pipeline()
        mlflow.log_params({"min_cycles": min_cycles, "target_seq_len": target_seq_len})

        f = CustomTransformer()
        mlflow.log_params({"input_dim": f.i_dim, "hidden_dim": f.h_dim})

        # -------- Optim parameters --------

        BATCH_SIZE = 32
        iters = len(X_train) // BATCH_SIZE

        train_indices = torch.randperm(len(X_train))

        lr = 1e-2
        epochs = 100
        optim = torch.optim.AdamW(f.parameters(), lr=lr) # used only for zero_grad()

        grad_clip = True
        max_grad_norm = 1.0

        losses, scores = [], []

        # parameters_names = list(n for n,_ in f.named_parameters())

        # alpha_mom = 0.8
        # momentum = dict(zip(parameters_names, [0]*len(parameters_names)))

        optim_params = {
            "batch_size": BATCH_SIZE,
            "learning_rate": lr,
            "epochs": epochs,
            "optimizer": optim.__class__,
            "grad_clip": grad_clip,
            "max_grad_norm": max_grad_norm,
        }
        mlflow.log_params(optim_params)

        # -------- Training loop --------

        os.makedirs("plots", exist_ok=True) # temp dir for plots

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
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(f.parameters(), max_grad_norm)

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
                
                mlflow.log_metrics(
                        {
                            "train_loss": mloss_train.item(),
                            "val_loss": loss_val.item(),
                            "train_rmse": mrmse_train.item(),
                            "val_rmse": rmse_val,
                        },
                        step=epoch+1
                    )

                # plot validation set distribution overlap
                sns.histplot(Y_test, label="true")
                sns.histplot(tensors_val["z"].squeeze(), label="pred")
                plt.legend()

                plot_path = f"plots/epoch_{epoch+1}.png"
                plt.savefig(plot_path)

                # plt.show()

                plt.close('all')

                # Log the plot
                mlflow.log_artifact(plot_path, artifact_path="plots")

        # Log the model
        in_sample = {"X": X_train[0].unsqueeze(0).numpy(), "y": Y_train[0].unsqueeze(0).numpy()}

        with torch.no_grad():
            out_sample = f(X_train[0].unsqueeze(0), Y_train[0].unsqueeze(0))
            out_sample = {"forward_tensors": {n:t.numpy() for n,t in out_sample[1].items()}}

        model_signature = infer_signature(in_sample, out_sample)

        mlflow.pytorch.log_model(f, name=f"model_epoch_{epoch+1}", signature=model_signature, step=epoch+1)
        
        # Log the source code
        mlflow.log_artifact("custom_rul_prediction.py", artifact_path="src")

        # Add tag with repo url
        mlflow.set_tag("GitHub Repo", "https://github.com/lucacareddu/Custom-Neural-Networks")

        try:
            shutil.rmtree("plots")
        except Exception as e:
            print(f"Could not delete temporary plots dir: {e}).")


if __name__ == "__main__":
    # train()
    print(infer())