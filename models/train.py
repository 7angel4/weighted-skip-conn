import torch
import torch.optim as optim
import typing
from utils.datasets import load_data
from models.eval import *
from models.models import *


def init_training(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, dataset = load_data(params['dataset'], data_only=False)
    params["n_classes"] = dataset.num_classes  # number of target classes
    params["input_dim"] = dataset.num_features  # size of input features
    
    model = set_model(params, device)
    model.param_init()
    
    optimiser = optim.Adam(model.parameters(), 
                           lr=params['lr'], 
                           weight_decay=params['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    
    return model, data, optimiser, loss_fn

def train_only(params: typing.Dict,
               cmp_by=DEFAULT_CMP_BY,
               report_per_period=1000,
               print_results=True):
    """
    Trains a node classification model and
    returns the trained model object.
    """
    model, data, optimiser, loss_fn = init_training(params)
    n_epochs = params['epochs']

    # variables for early stopping
    best_results = EvalResults(-1,-1)  # best validation results
    prev_loss = float('inf')
    consec_worse_epochs = 0  # number of consecutive epochs with degrading results
    # k: stop if epochs_dec_acc >= patience
    patience = params['max_patience']

    # standard training with backpropagation
    for epoch in range(n_epochs):
        model.train()
        optimiser.zero_grad()
        out = model(data.x, data.edge_index) # forward pass
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward() # backward pass
        optimiser.step()

        # evaluate on validation set
        results = evaluate(model, data, data.val_mask, cmp_by=cmp_by)

        # early stopping
        if results >= best_results:
            best_results = results
            consec_worse_epochs = 0
        else:
            consec_worse_epochs += 1

        # patience exceeded -> stop training
        if consec_worse_epochs >= patience:
            if print_results:
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Best results: {best_results}")
            break

        # print training progress
        if (epoch+1) % report_per_period == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}...")
            print(f"Loss: {loss};")
            print(f"Validation Results:\n{results}\n")

    return model, best_results


def train_and_tune(params, hyperparam, hyperparam_values, 
                   cmp_by=DEFAULT_CMP_BY,
                   report_per_period=1000,
                   print_results=True):
    """
    Trains the model and performs hyperparameter tuning.

    Args:
    - params: Training parameters.
    - hyperparam: Name of the hyperparameter to tune (if any).
    - hyperparam_values: Values to test for the hyperparameter (if any).
    - report_per_period: Frequency of training status reports.
    - metric: Metric to optimise during training; one of ["accuracy", "micro_f1", or "macro_f1"].

    Returns:
    - if hyperparam_values = None, a pair: (trained model, its training performance) - same as [train]
    - otherwise, a triple: (optimal trained model, ts training performance, its hyperparameter value)
    """
    best_hyperparam_val, best_model = None, None, 
    best_results = EvalResults(-1,-1)

    for val in hyperparam_values:
        params[hyperparam] = val
        model, results = train_only(params, cmp_by, 
                                    report_per_period, 
                                    print_results)
        if results > best_results:
            best_results = results
            best_hyperparam_val = val
            best_model = model

    return best_model, best_results, best_hyperparam_val


def train(params, hyperparam=None, 
          hyperparam_values=None, 
          cmp_by=DEFAULT_CMP_BY,
          report_per_period=1000,
          print_results=True):
    """
    Wrapper training function.
    Returns a triple: (model, training results, training hyperparameters)
    """
    model_name = params['model_name']
    if model_name not in MODEL_SPEC_HYPERPARAM: # train without tuning
        model, res = train_only(params, cmp_by, report_per_period, print_results)
        return model, res, params
    else: 
        model, res, hyperparam_val = train_and_tune(params, hyperparam, 
                                                    hyperparam_values, 
                                                    cmp_by, report_per_period, 
                                                    print_results)
        tuned_hyperparam = MODEL_SPEC_HYPERPARAM[model_name]
        params[tuned_hyperparam] = hyperparam_val
        return model, res, params
    

TRAIN_LAYERS = range(2,21,2)

def train_diff_layers_model(params,
                            layers=TRAIN_LAYERS,
                            cmp_by=DEFAULT_CMP_BY,
                            report_per_period=100,
                            print_results=True):
    model_name = params['model_name']
    hyperparam = MODEL_SPEC_HYPERPARAM.get(model_name, None)
    hyperparam_values = MODEL_HYPERPARAM_RANGE.get(model_name, None)
    
    layers_to_model = dict()
    layers_to_hyperparams = dict()
    for n in layers:
        curr_params = params.copy()
        curr_params['n_layers'] = n
        model, res, hyperparams = train(curr_params, hyperparam, hyperparam_values, 
                                        cmp_by, report_per_period, print_results)
        layers_to_model[n] =  model
        layers_to_hyperparams[n] = hyperparams

    return layers_to_model, layers_to_hyperparams


DEFAULT_TRAINING_PARAMS = {
    "lr": 0.01,  # learning rate
    "weight_decay": 0.0005,  # weight_decay
    "epochs": 400,  # number of total training epochs
    "max_patience": 5, # number of k for early stopping
    "hid_dim": 64, # hidden dimensions
    "init_res_weight": 0
}

def training_params(model_name, dataset_name, init_res_weight=0, n_layers=2):
    params = DEFAULT_TRAINING_PARAMS.copy()
    params['model_name'] = model_name
    params['dataset'] = dataset_name
    params['init_res_weight'] = init_res_weight
    params['n_layers'] = n_layers
    return params