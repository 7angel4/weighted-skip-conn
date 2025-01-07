from models.train import *
from utils.datasets import *
import json

def test(model, dataset_name):
    data = load_data(dataset_name, data_only=True)
    return evaluate(model, data, data.test_mask)

def train_and_test(params,
                   layers=TRAIN_LAYERS,
                   cmp_by=DEFAULT_CMP_BY,
                   report_per_period=1000,
                   print_results=True,
                   export_results=True,
                   dir='./results/'):
    dataset_name = params['dataset']
    model_name = params['model_name']
    layers_to_model, layers_to_hyperparams = train_diff_layers_model(params, 
                                                                     layers, 
                                                                     cmp_by,
                                                                     report_per_period,
                                                                     print_results)
    layers_to_results = dict()  # number of layers : test results
    for n, model in layers_to_model.items():
        layers_to_results[n] = test(model, dataset_name)
        layers_to_model[n] = model
        if export_results:
            torch.save(model.state_dict(), f'{dir}{dataset_name}/{n}_layers.pt')

    if print_results:
        print(f"\nTest results for {model_name} on {dataset_name}:")
        for n, results in layers_to_results.items():
            print(f"  {n}-layer model:")
            print(f"  Hyperparameter setting:")
            print(f"    {layers_to_hyperparams[n]}")
            print(f"    {results}")
            print()
    if export_results:
        with open(f'{dir}{dataset_name}/results.json', 'w') as fp:
            data = {n : metrics.to_serialisable() for n,metrics in layers_to_results.items() }
            json.dump(data, fp, indent=4)

    return layers_to_model, layers_to_results, layers_to_hyperparams


def training_params(model_name, dataset_name, init_res_weight=0, n_layers=2):
    params = DEFAULT_TRAINING_PARAMS.copy()
    params['model_name'] = model_name
    params['dataset'] = dataset_name
    params['init_res_weight'] = init_res_weight
    params['n_layers'] = n_layers
    return params