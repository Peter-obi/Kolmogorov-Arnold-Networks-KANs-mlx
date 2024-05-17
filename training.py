import argparse
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import mnist
from efficient_kan import KAN

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

def main(args):
    seed = 0
    num_layers = 2
    hidden_dim = 64
    num_classes = 10
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-3

    np.random.seed(seed)

    # Load the data
    train_images, train_labels, test_images, test_labels = map(
        mx.array, getattr(mnist, args.dataset)()
    )

    # Load the model
    model = KAN([28 * 28, hidden_dim, num_classes])
    mx.eval(model.parameters())

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)

    for e in range(num_epochs):
        tic = time.perf_counter()

        model.train()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        model.eval()
        accuracy = eval_fn(model, test_images, test_labels)

        toc = time.perf_counter()
        print(
            f"Epoch {e}: Test accuracy {accuracy.item():.3f},"
            f" Time {toc - tic:.3f} (s)"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train KAN on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="The dataset to use.",
    )
    args = parser.parse_args()

    if not args.gpu:
        mx.set_default_device(mx.cpu)

    main(args)