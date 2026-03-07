import wandb
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import main as train_main


sweep_config = {
    "method": "random",
    "metric": {
        "name": "validation_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {"values": [0.0005, 0.001, 0.005]},
        "optimizer":     {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "activation":    {"values": ["relu", "tanh", "sigmoid"]},
        "batch_size":    {"values": [32, 64, 128]},
        "num_layers":    {"values": [2, 3, 4]},
        "hidden_size":   {"values": [[64, 64], [128, 128], [128, 128, 128], [128, 64]]},
        "loss":          {"values": ["cross_entropy"]},
        "weight_init":   {"values": ["xavier"]},
        "weight_decay":  {"values": [0.0, 0.0001, 0.0005]},
        "epochs":        {"values": [10]},
        "dataset":       {"values": ["mnist"]}
    }
}


def train():
    """Called by wandb.agent for each sweep run."""

    with wandb.init() as run:
        config = wandb.config

        sys.argv = [
            "train.py",
            "-d",   config.dataset,
            "-e",   str(config.epochs),
            "-b",   str(config.batch_size),
            "-l",   config.loss,
            "-o",   config.optimizer,
            "-lr",  str(config.learning_rate),
            "-wd",  str(config.weight_decay),
            "-nhl", str(config.num_layers),
            "-a",   config.activation,
            "-w_i", config.weight_init,
            "--experiment", run.name,
            "-sz",
        ] + [str(h) for h in config.hidden_size]

        train_main()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="da6401-assignment1")
    wandb.agent(sweep_id, function=train, count=100)
