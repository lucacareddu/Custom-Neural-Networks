import random
import numpy as np
import torch

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

B = 1024
N = 4
M = 16

X = torch.randn((2*B,N,M))

Y_list = []
for sample in X:
    if (sample>1e-3).sum() > (sample<1e-3).sum():
        Y_list.append([1, 0])   # class 0
    else:
        Y_list.append([0, 1])   # class 1

Y = torch.tensor(Y_list)
# print(Y)

X_train, Y_train = X[:B], Y[:B]
X_val, Y_val = X[B:], Y[B:]

# -------- Parameters --------

D = 4
K = 2 # softmax with 2 classes (K=2) or sigmoid (K=1)

W1 = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((D,M))))
B1 = torch.nn.Parameter(torch.zeros((D,N)))
W2 = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.empty((K,D*N))))
B2 = torch.nn.Parameter(torch.zeros((K)))

# -------- Training functions --------

def forward(X, Y):
    pre_a1 = (X - X.mean()) / X.std()

    a1 = W1 @ pre_a1.mT + B1
    h1 = torch.relu(a1)

    pre_a2 = h1.flatten(start_dim=1)
    a2 = pre_a2 @ W2.T + B2 # == (w2 @ pre_a2.mT).mT

    if K > 1:
        s = torch.softmax(a2, dim=1)
    else:
        s = torch.sigmoid(a2)
        s = torch.concat([s, 1-s], dim=1)

    loss = - torch.mean(torch.sum(Y * torch.log(s), dim=1))

    return loss, s.detach(), pre_a2.detach(), h1.detach(), pre_a1

def backward(X, Y, s, pre_a2, h1):
    dL_a2 = s - Y if K > 1 else s[:,0,None] - Y[:,0,None]

    B2_grad = dL_a2
    W2_grad = B2_grad.unsqueeze(2) @ pre_a2.unsqueeze(1)

    B1_grad = (dL_a2 @ W2).reshape((B,D,N)) * (h1 > 0).float()
    W1_grad = B1_grad @ X

    return (W2_grad.mean(0), W1_grad.mean(0)), (B2_grad.mean(0), B1_grad.mean(0))


# -------- Training loop --------

lr = 1 if K > 1 else 1

for epoch in range(100):
    loss_train, s_train, pre_a2, h1, pre_a1 = forward(X_train, Y_train)
    
    if (epoch+1) % 1 == 0:
        acc_train = (Y_train == (s_train >= 0.5).int()).float().mean()
        print(f"Epoch {epoch+1} -> loss: {loss_train.item()}, acc: {acc_train.item()}")

    # Autograd
    loss_train.backward()

    with torch.no_grad():
        # Ours
        weights, biases = backward(pre_a1, Y_train, s_train, pre_a2, h1)
        
        # Ours vs Autograd
        # print(torch.allclose(W2.grad, weights[0], atol=1e-3))
        # print(torch.allclose(W1.grad, weights[1], atol=1e-3))
        # print(torch.allclose(B2.grad, biases[0], atol=1e-3))
        # print(torch.allclose(B1.grad, biases[1], atol=1e-3))

        # Gradient descent
        W2 -= lr * weights[0]
        W1 -= lr * weights[1]

        B2 -= lr * biases[0]
        B1 -= lr * biases[1]
        
        # zero out grads
        W2.grad = None
        W1.grad = None

        B2.grad = None
        B1.grad = None

        # Validation
        if (epoch+1) % 100 == 0:
            loss_val, s_val = forward(X_val, Y_val)[:2]
            acc_val = (Y_val == (s_val >= 0.5).int()).float().mean()
            print(f"    Eval - Epoch {epoch+1} - (Train/Val) -> loss: {loss_train.item()}/{loss_val.item()}, acc: {acc_train.item()}/{acc_val.item()}")
            