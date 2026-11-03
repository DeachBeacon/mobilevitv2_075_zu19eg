"""Paths shared by TF115 entry scripts; independent of current working directory."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
TF_ROOT = PROJECT / 'src/tensorflow'
TF_RECORDS = TF_ROOT / 'records'
TF_WEIGHTS = TF_ROOT / 'weights/mobilevitv2_075_10cls_relu_weights.h5'
INFERENCE = TF_ROOT / 'pb/inference/mobilevitv2_075_10cls_relu_inference.pb'
DEPLOY = TF_ROOT / 'pb/deploy'
REFERENCE_PB = TF_ROOT / 'pb/reference/mobilevitv2_075_10cls_relu_frozen.pb'
for path in (PROJECT / 'src/model_code/tf115',):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
