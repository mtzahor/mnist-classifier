# MNIST classifier

Trains and compares two PyTorch models on the [MNIST](http://yann.lecun.com/exdb/mnist/) digit dataset: a fully connected network with dropout and a small CNN. After training, it prints per-epoch loss and accuracy, then shows a figure with accuracy/loss curves and confusion matrices for both models.

## Requirements

- Python 3
- PyTorch with torchvision (CPU or CUDA)
- matplotlib, NumPy, scikit-learn

Install dependencies (adjust for your CUDA build if needed):

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

## Run

```bash
python mnist_classifier.py
```

The dataset is downloaded into `./data` on first run. Training uses CUDA when available, otherwise CPU.

## Models

- **LinearMNISTClassifier** — flatten → 128 → 64 → 10, ReLU and dropout between hidden layers.
- **CNNMNISTClassifier** — two conv/pool blocks, then dense layers to 10 classes.

Default training: 5 epochs, Adam (`lr=0.001`), cross-entropy loss, batch size 64 (train) / 1000 (test).
