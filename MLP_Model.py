#MLP Models
import torch
import torch.nn as nn
import os
from utilis_shorts import json_open


def activation_def(activation:str):
        '''
        Return the required activation function.
        Parameters:
            activation (str): Name of the activation function, e.g., 'Relu', 'Tanh'.
        Returns:
            nn.Module: The corresponding PyTorch activation function module.
        '''
        if activation in ('Relu', 'relu'):      #Rectified Linear Unit, sets negative values to zero. Often used for simplicity and computational efficiency.
            return nn.ReLU()
        elif activation in ('Tanh', 'tanh'):    #Hyperbolic tangent, maps input to the range [-1,1] Useful for symmetric outputs
            return nn.Tanh()
        elif activation in ('Sigmoid', 'sigmoid'):
            return nn.Sigmoid()
        elif activation in ('LeakyRelu', 'leakyrelu'): #A variant of ReLU that allows small gradients for negative values (controlled by a negative_slope parameter). Helps mitigate the "dying ReLU" problem.
            return nn.LeakyReLU()
        elif activation == 'Softplus': #Smooth approximation to ReLU. Outputs positive values with a small gradient for negative inputs
            return nn.Softplus()
        elif activation == 'Gelu': #Smooth approximation of ReLU using a Gaussian cumulative distribution. Often used in Transformer models
            return nn.GELU()
        elif activation == 'Swish': # SiLU is the PyTorch implementation of Swish #Sigmoid-weighted linear unit, where the output is x*sigmoid(x)
            return nn.SiLU()
        elif activation is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

class lin_layer(nn.Module):
    '''
    Initialize a linear layer with optional activation and dropout.
    Parameters:
        activation (str, optional): Name of the activation function to use.
        dropout_rate (float, optional): Dropout rate between 0 and 1. If False, dropout is not applied.
        bias (bool): Whether to include a bias term.
    '''
    def __init__(self, input_size:int, output_size:int, activation:str = None, dropout_rate: float = 0.0, bias=True) -> None:
        super().__init__()
        layer = [nn.Linear(input_size,output_size, bias)]
        if activation is not None:
            layer.append(activation_def(activation))
        if 0 < dropout_rate < 1:
            layer.append(nn.Dropout(p=dropout_rate))
        elif dropout_rate != 0:  # Check for invalid dropout rates that are not disabling dropout
            print(f"No dropout_used: {dropout_rate}. Dropout rate must be between 0 and 1.")
        self.lin_block =nn.Sequential(*layer)
    
    def forward(self,x):
        return self.lin_block(x)
    
class mlp_model(nn.Module):
    '''
    simple multi layer perceptron (mlp)
    layer-number: n
    hidden_size: n-1 
    activation: length=n
    dropout_list: length=n
    '''
    def __init__(self, layer_number:int, input_size:int,output_size:int,
                 hidden_size:list, activation: list , dropout_list:list, bias_list: list) -> None:
        super().__init__()
        self.layer_number = layer_number
        self.input_size = input_size
        #todo check order
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.act_list = activation
        self.dropout= dropout_list
        self.bias_list = bias_list

        layers=list()

        for i in range(self.layer_number):
            # a single-layer mlp must still map input_size -> output_size, so the
            # first/last cases are combined rather than checked in sequence
            is_first = (i == 0)
            is_last = (i == self.layer_number - 1)
            in_features = self.input_size if is_first else self.hidden_size[i-1]
            out_features = self.output_size if is_last else self.hidden_size[i]
            layers.append(lin_layer(in_features, out_features,
                          activation=self.act_list[i],
                          dropout_rate=0 if is_last else self.dropout,
                          bias= self.bias_list[i]))

        self.lin_layers= nn.Sequential(*layers)


    def forward(self,x):
        return self.lin_layers(x)
   
class LSTM_model(nn.Module):
    def __init__(self, lst_depth:int, layer_number:int, input_size:int,output_size:int,
                 hidden_size_lin:list, hidden_size_lstm:int, activation: list , dropout_list:list, bias_list: list) -> None:
        super().__init__()

        self.layer_number = layer_number
        self.input_size = input_size
        #todo check order
        self.hidden_lstm = hidden_size_lstm
        self.hidden_lin = hidden_size_lin
        self.output_size = output_size
        self.act_list = activation
        self.dropout_list= dropout_list
        self.bias_list = bias_list
        self.lstm_depth = lst_depth
        layers=list()
        if lst_depth >1:
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size= self.hidden_lstm, 
                            num_layers= lst_depth, dropout=self.dropout_list, bias= self.bias_list[0],
                            batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size= self.hidden_lstm,
                                bias= self.bias_list[0],batch_first=True)
        
        #linear layers
        self.mlp = mlp_model(layer_number=layer_number,input_size=self.hidden_lstm, hidden_size=self.hidden_lin,
                             output_size=output_size,dropout_list=dropout_list, activation=activation,
                             bias_list=bias_list)

        # Learnable initial hidden/cell state.  nn.LSTM silently defaults h0/c0 to
        # zeros, which asserts that every window begins with "no memory" — an
        # arbitrary prior, and a poor one for a model meant to carry accumulated
        # water deficit.  Costs 2 * depth * hidden parameters.  Initialised to zero,
        # so an untrained model reproduces the old behaviour exactly and only
        # departs from it as training moves them.
        self.h0 = nn.Parameter(torch.zeros(lst_depth, 1, self.hidden_lstm))
        self.c0 = nn.Parameter(torch.zeros(lst_depth, 1, self.hidden_lstm))

    def initial_state(self, batch_size:int):
        """Learnable (h0, c0) expanded to a batch. Contiguous, as nn.LSTM requires."""
        return (self.h0.expand(-1, batch_size, -1).contiguous(),
                self.c0.expand(-1, batch_size, -1).contiguous())

    def forward(self, x, hx=None, return_state:bool=False):
        """
        `hx` overrides the learnable initial state — pass the previous window's final
        state to carry memory across contiguous windows of the same site, and
        `return_state=True` to get that state back.  Doing so needs the loader to
        emit windows in order per site (and `.detach()` between windows to bound the
        backward graph); see the note in the LSTM section of the handoff.  Defaults
        keep the single-tensor signature the training loop calls.
        """
        if hx is None:
            hx = self.initial_state(x.size(0))
        out, state = self.lstm(x, hx)
        #if only one step in the future is predicted than
        out = out[:,-1,:]
        out = self.mlp(out)
        return (out, state) if return_state else out


class dag_structural_mlp_model(nn.Module):
    """
    Sequential node-wise DAG model.

    Observed root inputs are provided in a fixed order. Intermediate nodes and
    final targets are then computed one-by-one from their configured parent
    variables, so the DAG structure is part of the forward pass itself.
    """

    def __init__(
        self,
        layer_number: int,
        input_size: int,
        output_size: int,
        hidden_size: list,
        activation: list,
        dropout_list: list,
        bias_list: list,
        root_input_cols,
        node_order,
        node_to_parent_cols,
        target_order=None,
    ) -> None:
        super().__init__()
        self.layer_number = layer_number
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.act_list = activation
        self.dropout = dropout_list
        self.bias_list = bias_list
        self.root_input_cols = list(root_input_cols)
        self.node_order = list(node_order)
        self.node_to_parent_cols = {str(node): list(parents) for node, parents in node_to_parent_cols.items()}
        self.target_order = list(target_order or self.node_order)

        if len(self.root_input_cols) != int(input_size):
            raise ValueError(
                f"Input size ({input_size}) must match number of structural root inputs ({len(self.root_input_cols)})."
            )
        if len(self.target_order) != int(output_size):
            raise ValueError(
                f"Output size ({output_size}) must match number of structural targets ({len(self.target_order)})."
            )

        root_set = set(self.root_input_cols)
        self.heads = nn.ModuleDict()
        for node in self.node_order:
            parent_cols = self.node_to_parent_cols.get(node, [])
            if not parent_cols:
                raise ValueError(f"Structural DAG node '{node}' has no configured parents.")
            unknown = [parent for parent in parent_cols if parent not in root_set and parent not in self.node_order]
            if unknown:
                raise ValueError(f"Structural DAG node '{node}' has unknown parents: {unknown}")
            self.heads[node] = mlp_model(
                layer_number=layer_number,
                input_size=len(parent_cols),
                output_size=1,
                hidden_size=hidden_size,
                activation=activation,
                dropout_list=dropout_list,
                bias_list=bias_list,
            )

    def forward(self, x):
        values = {}
        for idx, col in enumerate(self.root_input_cols):
            values[col] = x[..., idx : idx + 1]

        for node in self.node_order:
            parent_tensors = [values[parent] for parent in self.node_to_parent_cols[node]]
            node_input = torch.cat(parent_tensors, dim=-1)
            values[node] = self.heads[node](node_input)

        return torch.cat([values[target] for target in self.target_order], dim=-1)


 

if __name__=='__main__':
    from Config_Files.config_spatial import build_config     ## explicitly set which experiment configuration file you wish to use
    
    config = build_config()
    save = config["cfg::save"]


    # #random intialized parameters
    # input_size = 7  # For example, 7 inputs ('Precip', 'Tair', 'PAR', 'VPD', 'fapar','C02', 'DOY')
    # output_size = 2  # For example, 2 outputs (GPP, ET)
    # hidden_size = [20, 15]  # Example hidden layer sizes
    # layer_number = 3  # Number of layers (first + process_layer + final)
    # activation = ['Relu', 'Relu', None]  # Example activations for each layer
    # dropout_list = [0.1, 0.1, 0.0]  # Dropout values
    # bias_list = [True, True, False]  # Bias for each layer

    
