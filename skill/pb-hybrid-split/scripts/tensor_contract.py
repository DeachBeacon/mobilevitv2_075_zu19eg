"""Portable PS/PL/PYNQ boundary audit. Does not program hardware or emulate rounding."""
import argparse
import json
from pathlib import Path
import numpy as np

def audit(value,contract):
    errors=[];expected=np.dtype(contract['host_dtype'])
    if value.shape!=tuple(contract['shape']):errors.append('shape mismatch')
    if value.dtype!=expected:errors.append('host dtype or byte order mismatch')
    if value.dtype.kind not in 'biuf':raise ValueError('Numeric tensor required')
    finite=bool(np.isfinite(value).all())
    if not finite:errors.append('non-finite values')
    report={'passed':False,'shape':list(value.shape),'dtype':str(value.dtype),'host_bytes':int(value.nbytes),
            'contiguous':bool(value.flags.c_contiguous),'finite':finite,'errors':errors,
            'scope':'Interface/range audit only; layout semantics, hardware rounding and accuracy need golden comparison'}
    if not value.flags.c_contiguous:errors.append('non-contiguous transfer array')
    if 'device_dtype' in contract:
        device=np.dtype(contract['device_dtype'])
        report.update(device_dtype=str(device),device_bytes=int(value.size*device.itemsize))
        if device!=expected and 'input_scale' not in contract:
            errors.append('host/device representation mapping is not defined')
    if 'input_scale' in contract:
        device=np.dtype(contract['device_dtype']);require=device.kind in 'iu'
        if not require:raise ValueError('Quantized device dtype must be integer')
        scale=float(contract['input_scale']);zero=int(contract.get('zero_point',0))
        if not np.isfinite(scale) or scale<=0:raise ValueError('input_scale must be positive and finite')
        limits=np.iinfo(device)
        if not limits.min<=zero<=limits.max:raise ValueError('Invalid zero point')
        low,high=(limits.min-zero)/scale,(limits.max-zero)/scale
        outside=int(np.count_nonzero((value<low)|(value>high))) if finite else int(value.size)
        report.update(representable_host_interval=[low,high],outside_elements=outside,
                      device_bytes=int(value.size*device.itemsize))
        if outside:errors.append('outside conservative quantized endpoint interval')
    report['passed']=not errors
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--tensor',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.report.exists():p.error('Choose a fresh report path')
    result=audit(np.load(str(a.tensor),allow_pickle=False),json.loads(a.contract.read_text(encoding='utf-8')))
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
