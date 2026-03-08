import sys
import os
import numpy as np
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import main as train_main


def run_train(params):
    """Build sys.argv and call train_main() directly (preserves wandb context)."""
    cmd = ["train.py"]
    for k, v in params.items():
        if isinstance(v, list):
            cmd.append(f"--{k}")
            cmd += [str(i) for i in v]
        else:
            cmd += [f"--{k}", str(v)]
    sys.argv = cmd
    train_main()


# ── Section 2.1 ── Data Exploration ────────────────────────────────────────────
def section_2_1():
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "data_exploration.py")])


# ── Section 2.2 ── Hyperparameter Sweep ────────────────────────────────────────
def section_2_2():
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "sweep.py")])


# ── Section 2.3 ── Optimizer Showdown (all 6 optimizers) ──────────────────────
def section_2_3():
    for opt in ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]:
        run_train({
            "optimizer":   opt,
            "activation":  "relu",
            "num_layers":  3,
            "hidden_size": [128, 128, 128],
            "epochs":      10,
            "experiment":  f"optimizer_{opt}"
        })


# ── Section 2.4 ── Vanishing Gradient Analysis ─────────────────────────────────
def section_2_4():
    for act in ["sigmoid", "relu"]:
        run_train({
            "activation":  act,
            "optimizer":   "adam",
            "num_layers":  4,
            "hidden_size": [128, 128, 128, 128],
            "epochs":      10,
            "experiment":  f"vanishing_{act}"
        })


# ── Section 2.5 ── Dead Neuron Investigation ───────────────────────────────────
def section_2_5():
    run_train({
        "activation":    "relu",
        "learning_rate": 0.1,
        "epochs":        10,
        "experiment":    "dead_relu"
    })
    run_train({
        "activation":    "tanh",
        "learning_rate": 0.1,
        "epochs":        10,
        "experiment":    "dead_tanh"
    })


# ── Section 2.6 ── Loss Function Comparison ────────────────────────────────────
def section_2_6():
    for loss in ["cross_entropy", "mean_squared_error"]:
        run_train({
            "loss":        loss,
            "optimizer":   "adam",
            "activation":  "relu",
            "hidden_size": [128, 128],
            "epochs":      10,
            "experiment":  f"loss_{loss}"
        })


# ── Section 2.7 ── Global Performance Analysis ─────────────────────────────────
def section_2_7():
    print(
        "Section 2.7: Open W&B dashboard → Charts → add 'train_accuracy' and "
        "'validation_accuracy' as an overlay across all sweep runs to identify overfitting."
    )


# ── Section 2.8 ── Error Analysis (Confusion Matrix) ──────────────────────────
def section_2_8():
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "inference.py")])


# ── Section 2.9 ── Weight Initialization & Symmetry ───────────────────────────
def section_2_9():
    for init in ["random", "xavier"]:
        run_train({
            "weight_init": init,
            "optimizer":   "adam",
            "activation":  "relu",
            "hidden_size": [128, 128],
            "epochs":      10,
            "experiment":  f"init_{init}"
        })


# ── Section 2.10 ── Fashion-MNIST Transfer Challenge ──────────────────────────
def section_2_10():
    configs = [
        {
            "dataset":     "fashion",
            "optimizer":   "adam",
            "activation":  "relu",
            "hidden_size": [128, 128, 128],
            "epochs":      15,
            "experiment":  "fashion_adam_relu"
        },
        {
            "dataset":     "fashion",
            "optimizer":   "momentum",
            "activation":  "tanh",
            "hidden_size": [128, 128, 128, 128],
            "epochs":      15,
            "experiment":  "fashion_momentum_tanh"
        },
        {
            "dataset":     "fashion",
            "optimizer":   "rmsprop",
            "activation":  "relu",
            "hidden_size": [128, 64, 64],
            "epochs":      15,
            "experiment":  "fashion_rmsprop_relu"
        }
    ]
    for cfg in configs:
        run_train(cfg)


def main():
    section_2_1()
    # section_2_2()  # already ran separately via sweep.py
    section_2_3()
    section_2_4()
    section_2_5()
    section_2_6()
    section_2_7()
    section_2_8()
    section_2_9()
    section_2_10()


if __name__ == "__main__":
    main()
