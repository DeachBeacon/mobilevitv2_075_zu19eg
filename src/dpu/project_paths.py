"""Paths for historical two-part analysis; final build is documented at root."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2]
ROOT=PROJECT/'src/dpu'
SOURCE=PROJECT/'src/tensorflow/pb/deploy/mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb'
FRONT=ROOT/'models/front_baseline/fp32/dpu_front_fp32.pb'
TAIL=ROOT/'models/tail_shared/ps_tail_fp32.pb'
NUMPY_GRAPH=PROJECT/'board/deploy_best_fused/model/ps_tail_numpy.json'
CALIB=PROJECT/'data/calibration/jpegimages'
CLE=ROOT/'models/front_cle'
SAMPLES=PROJECT/'data/samples/baseline'
SPLIT_MANIFEST=ROOT/'models/front_baseline/records/split_manifest.json'
def configure_imports():
    for path in (PROJECT/'board/deploy_best_fused/runtime',PROJECT/'src/dpu/split'):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
