import matplotlib.pyplot as plt


def plot_history(data):
    trn_loss = data['train_loss']
    val_loss = data['val_loss']

    fig, ax = plt.subplots(figsize=(5,3))
    plt.rcParams.update({'font.size': 11})
    ax.set_title('Loss evolution during training')
    ax.plot(trn_loss, c='blue', label='train loss')
    ax.plot(val_loss, c='red', label='validation loss')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    plt.tight_layout()
    plt.show()


def plot_fit_1d(x, y, x_fit, y_fit):
    fig, ax = plt.subplots(figsize=(5,4))
    plt.rcParams.update({'font.size': 11})
    ax.set_title('Results after training')
    ax.scatter(x, y, c='blue', label='dataset')
    ax.plot(x_fit, y_fit, c='red', lw=3, label='model fit')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_fit_2d(x, y, x_fit, y_fit):
    vmin, vmax = y.min(), y.max()

    fig, ax = plt.subplots(figsize=(5,4))
    plt.rcParams.update({'font.size': 11})
    ax.set_title('Results after training')
    ax.scatter(x[:,0], x[:,1], c=y, vmin=vmin, vmax=vmax, alpha=0.2, label='dataset')
    im = ax.scatter(x_fit[:,0], x_fit[:,1], c=y_fit, vmin=vmin, vmax=vmax, alpha=1, label='model fit')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()