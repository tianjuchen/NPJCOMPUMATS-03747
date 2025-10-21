import sys, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from aetor import AE  # Assuming you have a PyTorch version of the ae model


def encode_decode(model, x, fname, make_fig=False):
    # Convert numpy array to torch tensor and move to model device
    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x).float().to(device)

    # Encode
    with torch.no_grad():
        ld = model.encode(x_tensor)
    print("low dimensional data: ", ld.shape)

    if make_fig:
        # Decode
        with torch.no_grad():
            x_hat = model.decode(ld)
        print("reconstructed data: ", x_hat.shape)

        # Convert back to numpy for visualization
        x_hat = x_hat.cpu().numpy()[:100]
        x = x[:100]

        x_hat = np.reshape(x_hat, (x_hat.shape[0], x_hat.shape[1], x_hat.shape[2]))
        x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))

        res = 128
        X = np.linspace(0, 1, res)
        Y = np.linspace(0, 1, res)

        for t in range(100):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            ax1.set_aspect("equal")
            ax1.set_title("True Phase Field (t=%i)" % t, fontsize=22)
            cont1 = ax1.contourf(X, Y, x[t], 100, cmap="viridis", vmin=0, vmax=1)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(cont1, ax=ax1, cax=cax)
            cbar.ax.tick_params(labelsize=15)

            ax2.set_aspect("equal")
            ax2.set_title("Reconstructed Phase Field (t=%i)" % t, fontsize=22)
            cont2 = ax2.contourf(X, Y, x_hat[t], 100, cmap="viridis", vmin=0, vmax=1)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(cont1, ax=ax2, cax=cax)
            cbar.ax.tick_params(labelsize=15)

            plt.savefig(fname + "/t_" + str(t) + ".png")
            plt.close(fig)  # Close figure to free memory

    # Convert latent representation back to numpy
    return ld.cpu().numpy()


def main():
    address = "./../saved_models/ae_models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Loading data
    d = np.load("./../data/train_128_128.npz")
    x_train = d["X_func"]
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    )

    d = np.load("./../data/test_128_128.npz")
    x_test = d["X_func"]
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    # Loading model
    model = AE().to(device)
    model_number = np.load(address + "/best_ae_model_number.npy")
    model_address = (
        address + "/model_" + str(model_number) + ".pth"
    )  # Assuming .pth extension for PyTorch

    # Load model weights
    model.load_state_dict(torch.load(model_address, map_location=device))
    model.eval()  # Set model to evaluation mode

    batch_size = 1
    train_ld_ls = []
    test_ld_ls = []

    # Process training data
    for end in np.arange(batch_size, x_train.shape[0] + 1, batch_size):
        start = end - batch_size
        train_ld_ls.append(encode_decode(model, x_train[start:end], "train"))
        print("end: ", end)

    # Process test data
    for end in np.arange(batch_size, x_test.shape[0] + 1, batch_size):
        start = end - batch_size
        test_ld_ls.append(encode_decode(model, x_test[start:end], "test"))
        print("end: ", end)

    # Concatenate results
    train_ld = np.concatenate(train_ld_ls, axis=0)
    print("train_ld: ", train_ld.shape)
    np.savez("./../data/train_ld", X_func=train_ld)

    test_ld = np.concatenate(test_ld_ls, axis=0)
    print("test_ld: ", test_ld.shape)
    np.savez("./../data/test_ld", X_func=test_ld)

    print("Complete")


if __name__ == "__main__":
    main()
