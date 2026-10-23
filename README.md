# Custom Neural Networks 🧠

Personal project on custom neural networks with manual computing of the **back-propagation from scratch** (the math is provided in the pdf file).

- A custom $L$-layer **Vision Transformer Encoder**
- A custom 1-layer **Transformer Encoder** via class-token prediction
- A custom 2-layer **Feed-Forward Network**

| File | Description |
|---|---|
| `custom_anomaly_detection.py` | Masked-autoencoder ViT encoder on MVTec AD, scored by patch reconstruction error (image- and pixel-level AUROC) |
| `Custom_Anomaly_Detection.pdf` | Forward pass formalization and gradients derivation |

***Note:** The correctness of the computed gradients is assessed by using PyTorch AutoGrad as a reference.*