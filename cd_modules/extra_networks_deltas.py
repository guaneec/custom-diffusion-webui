from modules import extra_networks, sd_hijack
from cd_modules import deltas

# Swapping in and out weights for every Processing
# Not the most efficient method, might change later
class ExtraNetworkDelta(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('delta')
        self.backup = {}


    def activate(self, p, params_list):
        deltas.Delta.refresh()
        for params in params_list:
            params.items[1:] = params.items[1:] or [1]
            delta_name, strength = params.items
            strength = float(strength)
            delta = deltas.Delta(path=deltas.Delta.deltas[delta_name])
            delta.apply(strength, p.sd_model, self.backup)


    def deactivate(self, p):
        deltas.Delta.restore(p.sd_model, self.backup)
        self.backup = {}
