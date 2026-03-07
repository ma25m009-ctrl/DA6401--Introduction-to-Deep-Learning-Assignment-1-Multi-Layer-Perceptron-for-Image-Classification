# DA6401 Assignment 1 — NumPy MLP

A configurable, modular Multi-Layer Perceptron (MLP) built using only NumPy, trained on MNIST and Fashion-MNIST.

## Project Structure

```
│ README.md
│ requirements.txt
│
├───models
│     .gitkeep
│
└───src
    │ best_model.npy        ← saved automatically after training
    │ inference.py
    │ train.py
    │
    ├───ann
    │     activations.py
    │     neural_layer.py
    │     neural_network.py
    │     objective_functions.py
    │     optimizers.py
    │     __init__.py
    │
    └───utils
            data_loader.py
            __init__.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Run from the **project root**:

```bash
python src/train.py -d mnist -e 10 -b 64 -l cross_entropy -o adam \
  -lr 0.001 -wd 0.0 -nhl 3 -sz 128 128 128 -a relu -w_i xavier
```

### CLI Arguments

| Flag | Long | Description | Default |
|------|------|-------------|---------|
| `-d` | `--dataset` | `mnist` or `fashion` | `mnist` |
| `-e` | `--epochs` | Number of epochs | `10` |
| `-b` | `--batch_size` | Mini-batch size | `64` |
| `-l` | `--loss` | `cross_entropy` or `mean_squared_error` | `cross_entropy` |
| `-o` | `--optimizer` | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` | `sgd` |
| `-lr` | `--learning_rate` | Learning rate | `0.001` |
| `-wd` | `--weight_decay` | L2 regularization coefficient | `0.0` |
| `-nhl` | `--num_layers` | Number of hidden layers | `2` |
| `-sz` | `--hidden_size` | Neurons per hidden layer (space-separated) | `128 128` |
| `-a` | `--activation` | `sigmoid`, `tanh`, or `relu` | `relu` |
| `-w_i` | `--weight_init` | `random` or `xavier` | `xavier` |

Outputs saved to `src/`: `best_model.npy`, `config.json`

## Inference

```bash
python src/inference.py
```

Prints Accuracy, Precision, Recall, F1-score and saves `src/confusion_matrix.png`.

## Libraries Used

- `numpy` — all mathematical operations
- `keras` — dataset loading only (`keras.datasets`)
- `scikit-learn` — train/test split and evaluation metrics
- `matplotlib` — visualisation
- `wandb` — experiment tracking
