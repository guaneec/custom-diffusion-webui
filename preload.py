from modules.paths import models_path
import os

def preload(parser):
    parser.add_argument("--deltas-dir", type=str, help="finetuned delta weights directory", default=os.path.join(models_path, 'deltas'))
