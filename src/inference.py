import argparse
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Ensure src/ is on the path so ann/ and utils/ are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_layer import DenseLayer
from ann.activations import ReLU, Sigmoid, Tanh, Softmax
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

# Resolve paths relative to src/ (where best_model.npy lives)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def get_activation(name):
    if name == "relu":    return ReLU()
    if name == "sigmoid": return Sigmoid()
    if name == "tanh":    return Tanh()
    raise ValueError(f"Unknown activation: {name}")


def plot_confusion_matrix(cm, save_path):
    """Render confusion matrix using matplotlib only (no seaborn)."""

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    n_classes = cm.shape[0]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(range(n_classes))
    ax.set_yticklabels(range(n_classes))

    # Annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", color=color, fontsize=8)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Best Model", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference with a saved MLP model")
    parser.add_argument("--model_path",  default=os.path.join(SRC_DIR, "best_model.npy"),
                        help="Path to saved .npy weights (default: src/best_model.npy)")
    parser.add_argument("--config_path", default=os.path.join(SRC_DIR, "config.json"),
                        help="Path to saved config.json (default: src/config.json)")
    parser.add_argument("--output_dir",  default=SRC_DIR,
                        help="Directory to write metrics JSON and confusion matrix")
    return parser.parse_args()


def main():

    args = parse_arguments()

    with open(args.config_path) as f:
        config = json.load(f)

    weights = np.load(args.model_path, allow_pickle=True)

    dataset_name = "fashion_mnist" if config.get("dataset") == "fashion" else "mnist"
    _, _, _, _, X_test, y_test_oh, y_test = load_data(dataset_name)

    # Rebuild model architecture from saved config
    layers, activations = [], []
    prev = 784

    for h in config["hidden_size"]:
        layers.append(DenseLayer(prev, h))
        activations.append(get_activation(config["activation"]))
        prev = h

    layers.append(DenseLayer(prev, 10))
    activations.append(Softmax())

    model = NeuralNetwork(layers, activations, None, None)

    # Restore saved weights
    idx = 0
    for layer in model.layers:
        layer.W = weights[idx]
        layer.b = weights[idx + 1]
        idx += 2

    preds = model.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds, average="macro", zero_division=0)
    f1   = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics as JSON
    metrics = {
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1_score":  float(f1),
    }
    with open(os.path.join(args.output_dir, "inference_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm, os.path.join(args.output_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
