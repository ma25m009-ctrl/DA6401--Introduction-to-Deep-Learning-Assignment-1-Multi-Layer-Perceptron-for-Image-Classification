import numpy as np
import wandb
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_data


def main():

    wandb.init(project="da6401-assignment1", name="data_exploration")

    X_train, y_train, X_val, y_val, X_test, y_test_oh, y_test = load_data("mnist")

    labels = np.argmax(y_train, axis=1)

    table = wandb.Table(columns=["class", "image"])

    for digit in range(10):
        idx = np.where(labels == digit)[0][:5]
        for i in idx:
            img = X_train[i].reshape(28, 28)
            table.add_data(digit, wandb.Image(img))

    wandb.log({"mnist_samples": table})
    wandb.finish()


if __name__ == "__main__":
    main()
