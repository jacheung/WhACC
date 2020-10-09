import pathlib
import whacc

PACKAGE_ROOT = pathlib.Path(whacc.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'model_checkpoints/'
DATASET_DIR = PACKAGE_ROOT / 'datasets/'
VERSION = '0.0.10'