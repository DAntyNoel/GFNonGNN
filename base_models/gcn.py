import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from utils import get_logger

logger = get_logger('GCN')

class GCN(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self._params = params
        self.in_channels = params.in_channels
        self.hidden_channels = params.hidden_channels
        self.out_channels = params.out_channels
        self.use_gdc = params.use_gdc
        self.num_layers = params.num_layers

        self.gfn_sample_result_path = params.gfn_sample_result_path
        self.save_gfn_sample_result = params.save_gfn_sample_result
        self.save_num_edges = params.save_num_edges

        self.now_epoch = None
        self.out_conv = GCNConv(self.hidden_channels, self.out_channels,
                             normalize=not self.use_gdc)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.in_channels, self.hidden_channels,
                             normalize=not self.use_gdc))
        for _ in range(self.num_layers - 2):
            self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels,
                                      normalize=not self.use_gdc))

    def forward(self, x, edge_index, GFN=None, start_layer=-1):
        if GFN is not None:
            edge_indexs, (states_fin, values, log_rs) = GFN.sample(x, edge_index, self.num_layers - 1, return_edge_logits=True)
            if self.save_gfn_sample_result:
                # Save the GFN sample result
                result = torch.load(self.gfn_sample_result_path)
                states_fin = states_fin[:, :self.save_num_edges].unsqueeze(0)
                values = values.unsqueeze(0)
                if result['states_fin'] is None:
                    result['states_fin'] = states_fin
                    result['values'] = values
                else:
                    result['states_fin'] = torch.cat((result['states_fin'], states_fin), dim=0)
                    result['values'] = torch.cat((result['values'], values), dim=0)
                torch.save(result, self.gfn_sample_result_path)
                
                
        for i, conv in enumerate(self.convs):
            if GFN is not None:
                if self._params.is_debug:
                    assert torch.all(edge_indexs[i] == edge_index)
                x = conv(x, edge_indexs[i]).relu()
            else:
                x = conv(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            if i > start_layer > 0: 
                # start_layer is the layer to train early GNN, so that
                # the GFN can be trained with the early GNN features.
                break
        x = self.out_conv(x, edge_index)
        return x

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Add now_epoch to the state_dict
        state_dict[prefix + 'now_epoch'] = self.now_epoch
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.now_epoch = state_dict.get('now_epoch', None)
        # Remove now_epoch from the state_dict
        if 'now_epoch' in state_dict:
            del state_dict['now_epoch']
        super().load_state_dict(state_dict, strict)
        
