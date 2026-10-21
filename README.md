# NN function interpolator

A small and lightweight Python library for function approximation and interpolation using **feedforward neural networks (FFNN)** built with PyTorch. It has been build to test the *universal approximation theorem*, which, in this particular case, states that any FFNN with at least one hidden layer and non-polynomial activation functions (e.g. ReLU) could, in principle, approximate any continuous function to any desired degree of accuracy.

The import name of the library is `nninterpolator` and provides a simple workflow for loading tabular data, training a neural network with it and creating a continuous fit.

## Features

- **Neural network regression:** Fully connected neural networks with ReLU activation functions in hidden layers and a linear output layer.
- **SGD optimizer**: Manual gradient descent updates with gradient-norm clipping.
- **Training diagnostics:** Track mean squared error (MSE) on training and validation data across epochs.
- **CSV data loading:** Select input and output columns directly from a CSV file.
- **Training and validation sets:** Configurable split ratio, shuffling, and batch sizes.
- **Automatic standardization:** Inputs and targets are standardized during training and predictions are transformed back to the original target scale.
- **Visualization:** Plot loss evolution and fitted functions for supported one- and two-dimensional inputs.

## Installation

Clone the repository:

```bash
git clone https://github.com/Arnau-V-B/NN-function-interpolator.git
```

Install the project and its dependencies:

```bash
pip install local_path/NN-function-interpolator
```
*Where `local_path/` is the local directory where the repository has been cloned*.

The library uses **PyTorch**, **pandas**, and **Matplotlib**. They should be installed automatically at the same time as the library is installed, as specified in the [pyproject.toml](./pyproject.toml) project installation configuration file. If this is not the case, they can otherwise be installed from the [requirements.txt](./requirements.txt) file with the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
nn-interpolator/
├── examples/
│   ├── example1a.csv
│   ├── example1b.csv
│   └── example2.csv
├── notebooks/
│   └── getting_started.ipynb
│   └── model_tuning_deep-dive.ipynb
├── src/
│   └── nninterpolator/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       └── plotting.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

- `__init__.py`: Exposes `NNInterpolator` and `Loader` as the package's public imports.
- `data.py`: Implements CSV loading, train/validation splitting, and standardization utilities.
- `model.py`: Implements neural network initialization, training, loss tracking, and prediction.
- `plotting.py`: Implements loss-history and model-fit visualizations.
- `examples/`: Contains example CSV datasets.
- `notebooks/`: Contains detailed notebooks on how to get started with the library.

## Quick Start

The following example demonstrates how to load a dataset, train a neural network and make predictions. For a more in-depth workflow see [getting_started.ipynb](./notebooks/getting_started.ipynb), and for a detailed explanation of each parameter see [model_tuning_deep-dive.ipynb](./notebooks/model_tuning_deep-dive.ipynb).

The dataset inside [example1a.csv](./examples/example1a.csv) contains two columns: the input variable \(x\) and the corresponding target \(y\).

```python
import torch
from nninterpolator import Loader, NNInterpolator

# Load the dataset.
loader = Loader(
    trn_perc=0.8,
    batch_sz=32,
    shuffle=True,
)

dls = loader.from_csv(
    path="examples/example1a.csv",
    x_cols=[0],
    y_cols=[1],
)

# Initialize the neural network.
model = NNInterpolator(
    nlay=2,
    lay_sz=32,
)

# Train the model.
history = model.fit(
    dls,
    lr=1e-3,
    nepochs=1000,
    plot=True,
)

# Predict at new input values.
x_new = torch.linspace(0, 1, 100).reshape(-1, 1)
y_pred = model.predict(x_new)

print(y_pred)
```

Adjust the input range to match the domain of your dataset. The example assumes that the input variable lies between 0 and 1.

## Limitations

- The current implementation targets only regression through interpolation. Extrapolation the dataset outside its domain is not possible.
- The network architecture uses fully connected layers and ReLU hidden-layer activations. Changing the activation functions requires manual editing of the code.
- The model is trained using a basic manual gradient-descent update rule rather than a PyTorch built-in optimizer.
- Plotting is limited to one- or two-dimensional inputs with a single output.

## License

This is just a small personal project to learn about the implementation of neural networks, so obviously it has no license and it can be freely used and modified by any user.