import os
import glob
from modules.paths import models_path
from safetensors.torch import safe_open, save_file
from cd_modules.compression import decompose
import torch
import json

class Delta(dict):
    deltas = {}

    @classmethod
    def refresh(cls):
        cls.deltas = cls.list_deltas()
    
    @staticmethod
    def list_deltas():
        res = {}
        for filename in sorted(
            glob.iglob(
                os.path.join(models_path, "deltas", "**/*.safetensors"), recursive=True
            )
        ):
            name = os.path.splitext(os.path.basename(filename))[0]
            res[name] = filename
        return res

    @staticmethod
    def restore(model, backup):
        with torch.no_grad():
            for k, v in model.named_parameters():
                if k in backup:
                    v[:] = backup[k]
 
        
    def __init__(self, *, path=None, tensors=None):
        assert (path is None) != (tensors is None)
        if path is not None:
            st = safe_open(path, 'pt')
            metadata = json.loads(st.metadata()['json'])
            entries = metadata['entries']
            self.metadata = metadata['meta']
            def get_entry(k):
                if entries[k] == 'delta':
                    d = st.get_tensor(k)
                elif entries[k] == 'delta_factors':
                    d = st.get_tensor(k+'.US').float() @ st.get_tensor(k+'.Vh').float()
                else:
                    raise ValueError(f'Unknown format: {entries[k]}')
                return d
            self.update(((k, get_entry(k)) for k in entries))
        else:
            self.update(tensors)
            self.metadata = {'version': '0.2.0'}
        
    def apply(self, strength, model, backup):
        for k, v in model.named_parameters():
            if k not in self:
                continue
            if k not in backup:
                backup[k] = v.detach().clone()
            with torch.no_grad():
                v[:] = v.detach() + self[k].to(v.device) * strength

    def save(self, path, fmt='delta', top_sum=None):
        metadata = {'meta': self.metadata, 'entries': {k: fmt for k in self}}
        if fmt == 'delta':
            tensors = self
        elif fmt == 'delta_factors':
            tensors = {}
            for k, v in self.items():
                tensors[k+'.US'], tensors[k+'.Vh'] = map(lambda a: a.half().contiguous(),
                decompose(v, top_sum))
        else:
            raise ValueError(f'unknown storage format: {fmt}')
        save_file(tensors, path, {'json': json.dumps(metadata)})

