import modules.scripts as scripts
import gradio as gr
from modules.ui import create_refresh_button
from modules.paths import models_path
import modules.sd_models as sd_models
import glob
import os
from safetensors.torch import load_file
import torch
from math import prod

pluggables = {}
backup = {}


def list_pluggables():
    res = {}
    for filename in sorted(
        glob.iglob(os.path.join(models_path, 'pluggables', "**/*.safetensors"), recursive=True)
    ):
        name = os.path.splitext(os.path.basename(filename))[0]
        if name != "None":
            res[name + f"({sd_models.model_hash(filename)})"] = filename
    return res


last_model_hash, last_pluggable_name = None, None


def apply(pluggable_name, model):
    global backup, last_model_hash, last_pluggable_name
    if (last_model_hash, last_pluggable_name) == (model.sd_model_hash, pluggable_name):
        return
    if last_model_hash != model.sd_model_hash:
        backup = {}
    d = load_file(pluggables[pluggable_name]) if pluggable_name != "None" else {}
    print(f"loaded weights ({sum(prod(p.shape) for p in d.values())} params)")
    for k, v in model.model.named_parameters():
        if k in d and k not in backup:
            backup[k] = v.detach().clone()
        if k in d:
            with torch.no_grad():
                v[:] = d[k]
            print(f"loaded {k} from {pluggable_name}")
        elif k in backup:
            with torch.no_grad():
                v[:] = backup[k]
            del backup[k]
            print(f"restored {k}")
    last_model_hash, last_pluggable_name = model.sd_model_hash, pluggable_name


class Script(scripts.Script):
    def title(self):
        return "Custom Diffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Row():
            pluggable_name = gr.Dropdown(
                ["None", *pluggables.keys()], label="Pluggable weights", value="None"
            )

            def refresh():
                global pluggables
                pluggables = list_pluggables()

            refresh()
            refresh = create_refresh_button(
                pluggable_name,
                refresh,
                (lambda: dict(choices=["None", *pluggables.keys()])),
                "plug_refresh",
            )
        return [pluggable_name, refresh]

    def process(self, p, pluggable_name, _refresh):
        apply(pluggable_name, p.sd_model)
