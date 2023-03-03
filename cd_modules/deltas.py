import os
import glob
from modules.paths import models_path
from safetensors.torch import safe_open, save_file
from cd_modules.compression import decompose
import torch
import json
from modules.sd_hijack import model_hijack
from modules import shared

class Delta:
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
        torch.cuda.empty_cache()
 
        
    def __init__(self, *, path=None, tensors=None, embeddings=None):
        assert (path is None) != (tensors is None)
        from modules.textual_inversion.textual_inversion import Embedding
        self.embeddings = embeddings or {}
        if path is not None:
            st = safe_open(path, 'pt')
            if 'json' in st.metadata():
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
                self.entries = {k: get_entry(k) for k in entries}
            else:
                tuner_meta = json.loads(st.metadata()['tuner'])
                entries = tuner_meta['weights']
                def get_entry(k):
                    if entries[k] == 'delta':
                        d = st.get_tensor(f"weights/{k}")
                    elif entries[k] == 'delta_factors':
                        d = st.get_tensor(f"weights/{k}.US").float() @ st.get_tensor(f"weights/{k}.Vh").float()
                    else:
                        raise ValueError(f'Unknown format: {entries[k]}')
                    return d
                self.entries = {k: get_entry(k) for k in entries}
                self.metadata = {}
                self.embeddings = {k: Embedding(st.get_tensor(f"embeddings/{k}").to(shared.devices.device), k) for k in tuner_meta['embeddings']}
        else:
            self.entries = {**tensors}
            self.metadata = {'version': '0.2.0'}
        
        for k, v in self.embeddings.items():
            model_hijack.embedding_db.register_embedding(v, shared.sd_model)
            print("added embedding: ", k)
        
    def apply(self, strength, model, backup):
        for k, v in model.named_parameters():
            if k not in self.entries:
                continue
            if k not in backup:
                backup[k] = v.detach().clone()
            with torch.no_grad():
                v[:] = v.detach() + self.entries[k].to(v.device) * strength

    def save(self, path, fmt='delta', top_sum=None):
        metadata = {'meta': self.metadata, 'entries': {k: fmt for k in self.entries}}
        if fmt == 'delta':
            tensors = self.entries
        elif fmt == 'delta_factors':
            tensors = {}
            for k, v in self.entries.items():
                tensors[k+'.US'], tensors[k+'.Vh'] = map(lambda a: a.half().contiguous(),
                decompose(v, top_sum))
        else:
            raise ValueError(f'unknown storage format: {fmt}')
        save_file(tensors, path, {'json': json.dumps(metadata)})

    @property
    def step(self):
        return self.metadata.get('step', 0)

    @step.setter
    def step(self, step: int):
        self.metadata['step'] = step

