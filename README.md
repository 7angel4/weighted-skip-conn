# Weighted Skip Connections

This project explores the use of Weighted Skip Connections in graph neural networks (GNNs), as introduced in the submitted paper for the MT2024-25 Graph Representation Learning Exam.

This repository contains the model implementations, the original experiments conducted in `jupyter notebook`, plus a CLI for users to interact with the implemented models.
The CLI allows users to customise various parameters for model training and testing.

## Installation

To run this experiment, ensure you have the following dependencies installed:

1. **Python (3.7 or higher)**: This project is compatible with Python 3.7 or later.
2. **Required Python Packages**: These packages can be installed using `pip`.

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface (CLI)

The experiment can be run using the command line. 
Below is the basic usage pattern:

```bash
python run.py --config <path_to_config_file> [other optional arguments]

Where <path_to_config_file> refers to the path of a YAML configuration file containing default parameters. 
You can specify any additional command-line arguments to override specific configurations.
We provide a sample configuration at: ./default_config.yaml
```

### CLI Arguments

**General Arguments:**

- `--config <path_to_config_file>`: (Required) Path to a YAML configuration file that holds the model, training, and experimental parameters.

The following arguments will override the default parameter in the config file.

**Arguments for model_config:**

- `--model_name <model_name>`: Name of the model (e.g., one of ['WSkipGNN', 'PlainGNN', 'SkipGNN', 'JumpKGNN']). This will override the default `model_name` in the configuration.
- `--init_res_weight <value>`: Initial weight for residual connections (defaults to 0 in the config).

**Arguments for train_config:**

- `--dataset <dataset_name>`: The dataset name to use for training (e.g., Cora). 
- `--lr <learning_rate>`: Learning rate for the optimizer (e.g., 0.01). 
- `--weight_decay <weight_decay>`: Weight decay for regularization (e.g., 0.0005).
- `--epochs <num_epochs>`: Number of epochs to run the training (e.g., 400).
- `--max_patience <max_patience>`: Maximum consecutive number of epochs for which validation accuracy may decrease before training stops (e.g., 5).

**Arguments for experiment_config:**

- `--layers <layer_list>`: A comma-separated list of integers representing the layer numbers to train and test (e.g., "2,4" will train and test a 2-layer model and a 4-layer model).
- `--report_per_period <int>`: How frequently (in epochs) the training progress is reported.
- `--export`: If specified, the results will be exported (default behavior defined in config).
- `--dir`: Destination directory for exported results, if export is true (overrides config).


### Example Usage
1. Run the experiment with a **specific configuration file**:

```bash
python run.py --config config.yaml
```

2. **Overriding parameters** with the CLI:

```bash
python run.py --config config.yaml --model_name "WSkipGNN" --dataset "Cora" --epochs 500 --report_per_period 100
```

This example overrides the model name to WSkipGNN, sets the dataset to Cora, adjusts the number of epochs to 500, and reports every 100 epochs.


## Repository structure

**For CLI:**

- `models`: Python module for model implementations, training and testing models
- `utils`: Python module for miscellaneous utility functions
- `run.py`: Entry point for the program
- `default_config.yaml`: Sample config to be passed to the CLI
- `requirement.txt`: A list of dependencies required for the CLI

**Our own experiments:**

- `data`: datasets for training the models
- `results`: our experimental results (visualisations, saved models for reproduction, csv output from evaluation...)
- `jupyter-notebook`: Jupyter notebooks where our original experiments are conducted
