import os
import glob
from modules.paths import models_path

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
        res[name] = filename
    return res
