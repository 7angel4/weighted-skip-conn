import torch
from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as datasets

from torch_geometric.nn import GCNConv
import numpy as np

MODELS = ['PlainGNN', 'SkipGNN', 'JumpKGNN', 'WSkipGNN']
MODEL_SPEC_HYPERPARAM = { 'WSkipGNN': 'init_res_weight' }
MODEL_HYPERPARAM_RANGE = { 'WSkipGNN': np.arange(-1,1.1,0.2) }


class WSkipGNN(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int,
                 n_classes: int, n_layers: int,
                 init_res_weight: float = 0.3,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        """
        Args:
          input_dim: input feature dimension
          hid_dim: hidden feature dimension
          n_classes: number of target classes
          n_layers: number of layers
        """
        super(WSkipGNN, self).__init__()
        self.n_layers = n_layers
        self.res_weight = nn.Parameter(torch.tensor(init_res_weight,
                                                    dtype=torch.float32))

        layers = [GCNConv(input_dim, hid_dim)]
        layers += [GCNConv(hid_dim, hid_dim) for _ in range(1, n_layers)]
        self.layers = nn.ModuleList(layers)
        self.mlp = nn.Linear(hid_dim, n_classes)
        self.act_fn = act_fn
        self.param_init()

    def forward(self, X, A) -> torch.Tensor:
        X = self._forward_before_final_layer(X, A)
        return self.mlp(X)  # MLP maps to logits

    def generate_node_embeddings(self, X, A) -> torch.Tensor:
        """Generate node embeddings without applying the MLP."""
        X = self._forward_before_final_layer(X, A)
        return self.act_fn(X)

    def _forward_before_final_layer(self, X, A) -> torch.Tensor:
        """ Forward through all layers, 
            without applying the activation and MLP after the final layer. 
        """
        X = self.layers[0](X, A)
        X = self.act_fn(X)

        for l in self.layers[1:-1]:
            residual = X  # previous layer's representation
            X = l(X, A)
            X = self.act_fn(X)
            X = X + self.res_weight * residual  # Add weighted residual connection

        return self.layers[-1](X, A)

    def param_init(self):
        nn.init.xavier_uniform_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)
        for conv in self.layers:
            nn.init.xavier_uniform_(conv.lin.weight)
            nn.init.zeros_(conv.bias)


class PlainGNN(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int,
                 n_classes: int, n_layers: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        """
        Args:
            input_dim: input feature dimension
            hid_dim: hidden feature dimension
            n_classes: number of target classes
            n_layers: number of layers
        """
        super(PlainGNN, self).__init__()
        # assert n_layers > 1
        self.n_layers = n_layers

        layers = [GCNConv(input_dim, hid_dim)]
        layers += [GCNConv(hid_dim, hid_dim) for _ in range(1, n_layers)]
        self.layers = nn.ModuleList(layers)
        self.mlp = nn.Linear(hid_dim, n_classes) # final MLP for generating logits
        self.act_fn = act_fn

        self.param_init()

    def forward(self, X, A) -> torch.Tensor:
        X = self._forward_before_final_layer(X, A)
        return self.mlp(X)  # do not apply non-linearity before MLP

    def generate_node_embeddings(self, X, A) -> torch.Tensor:
        """ Generate node embeddings without applying the MLP. """
        X = self._forward_before_final_layer(X, A)
        X = self.act_fn(X)
        return X  # raw GNN output without applying MLP

    def _forward_before_final_layer(self, X, A) -> torch.Tensor:
        """ Apply all layers except for the final layer and MLP. """
        for l in self.layers[:-1]: # message-passing through all layers except for the last
            X = l(X, A)
            X = self.act_fn(X)
        return self.layers[-1](X, A)  # raw GNN output without applying MLP

    def param_init(self):
        # initialise MLP parameters
        nn.init.xavier_uniform_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)
        for conv in self.layers:
            # initialise weight in each layer's Linear object
            nn.init.xavier_uniform_(conv.lin.weight)
            nn.init.zeros_(conv.bias)


class SkipGNN(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int,
                 n_classes: int, n_layers: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        """
        Args:
          input_dim: input feature dimension
          hid_dim: hidden feature dimension
          n_classes: number of target classes
          n_layers: number of layers
          act_fn: activation function
        """
        super(SkipGNN, self).__init__()
        assert n_layers > 1
        self.n_layers = n_layers

        layers = [GCNConv(input_dim, hid_dim)]
        layers += [GCNConv(hid_dim, hid_dim) for _ in range(1, n_layers)]
        self.layers = nn.ModuleList(layers)
        self.mlp = nn.Linear(hid_dim, n_classes)
        self.act_fn = act_fn

        self.param_init()

    def forward(self, X, A) -> torch.Tensor:
        X = self._forward_before_final_layer(X, A)
        return self.mlp(X)

    def generate_node_embeddings(self, X, A) -> torch.Tensor:
        """Generate node embeddings without applying the MLP."""
        X = self._forward_before_final_layer(X, A)
        return self.act_fn(X)

    def _forward_before_final_layer(self, X, A) -> torch.Tensor:
        """ Forward through all layers, 
            without applying the activation and MLP after the final layer. 
        """
        X = self.layers[0](X, A)
        X = self.act_fn(X)

        for l in self.layers[1:-1]:
            residual = X
            X = l(X, A)
            X = self.act_fn(X)
            X = X + residual  # add skip connection

        return self.layers[-1](X, A)

    def param_init(self):
        nn.init.xavier_uniform_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)
        for conv in self.layers:
            nn.init.xavier_uniform_(conv.lin.weight)
            nn.init.zeros_(conv.bias)



class JumpKGNN(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int,
                 n_classes: int, n_layers: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        """
        Args:
            input_dim: input feature dimension
            hid_dim: hidden feature dimension
            n_classes: number of target classes
            n_layers: number of layers
            act_fn: activation function
        """
        super(JumpKGNN, self).__init__()
        assert n_layers > 1
        self.n_layers = n_layers

        layers = [GCNConv(input_dim, hid_dim)]
        layers += [GCNConv(hid_dim, hid_dim) for _ in range(1, n_layers)]
        self.layers = nn.ModuleList(layers)
        self.mlp = nn.Linear(hid_dim, n_classes)
        self.act_fn = act_fn

        self.param_init()


    def _layer_outputs(self, X, A) -> torch.Tensor:
        """ Outputs of all layers
            (no activation applied to the final layer -
            i.e. last element is just logits)
        """
        outputs = []
        for l in self.layers[:-1]:
            X = l(X, A)
            X = self.act_fn(X)
            outputs.append(X)

        outputs.append(self.layers[-1](X, A))
        return outputs

    
    def forward(self, X, A) -> torch.Tensor:
      return self._layer_outputs(X, A)[-1]

    def generate_node_embeddings(self, X, A) -> torch.Tensor:
        outputs = self._layer_outputs(X, A)[:-1]
        return torch.max(torch.stack(outputs), 0)[0]  # max pooling

    def param_init(self):
        for conv in self.layers:
            nn.init.xavier_uniform_(conv.lin.weight)
            nn.init.zeros_(conv.bias)


def set_model(params, device):
    """
    Returns the model initialised based on the configuration specified by `params`.
    """
    model_name = params['model_name']
    model_class = globals().get(model_name)

    if model_class is None or model_name not in MODELS:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")

    # Dynamically pass only the relevant arguments by filtering params
    required_keys = model_class.__init__.__annotations__.keys()
    model_args = {key: params[key] for key in required_keys if key in params}

    return model_class(**model_args).to(device)