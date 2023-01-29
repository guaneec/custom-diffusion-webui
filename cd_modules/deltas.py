import os
import glob
from modules.paths import models_path
import modules.sd_models as sd_models

deltas = {}

def refresh():
    global deltas
    deltas = list_deltas()


def list_deltas():
    res = {}
    for filename in sorted(
        glob.iglob(
            os.path.join(models_path, "deltas", "**/*.safetensors"), recursive=True
        )
    ):
        name = os.path.splitext(os.path.basename(filename))[0]
        if name != "None":
            res[name + f"({sd_models.model_hash(filename)})"] = filename
    return res
