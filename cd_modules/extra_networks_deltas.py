from modules import extra_networks
from safetensors import safe_open
import torch
from math import prod
import json
from cd_modules import deltas

# Swapping in and out weights for every Processing
# Not the most efficient method, might change later
class ExtraNetworkDelta(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('delta')
        self.backup = {}


    def activate(self, p, params_list):
        model = p.sd_model
        deltas.refresh()
        for params in params_list:
            params.items[1:] = params.items[1:] or [1]
            delta_name, strength = params.items
            strength = float(strength)
            st = safe_open(deltas.deltas[delta_name], 'pt')
            metadata = json.loads(st.metadata()['json'])
            entries = metadata['entries']
            for k, v in model.named_parameters():
                if k not in entries:
                    continue
                if k not in self.backup:
                    self.backup[k] = v.detach().clone()
                with torch.no_grad():
                    if entries[k] == 'delta':
                        d = st.get_tensor(k)
                    elif entries[k] == 'delta_factors':
                        d = st.get_tensor(k+'.US').float() @ st.get_tensor(k+'.Vh').float()
                    else:
                        raise ValueError(f'Unknown format: {entries[k]}')
                    v[:] = v.detach() + d.to(v.device) * strength


    def deactivate(self, p):
        model = p.sd_model
        with torch.no_grad():
            for k, v in model.named_parameters():
                if k in self.backup:
                    v[:] = self.backup[k]
        self.backup = {}
