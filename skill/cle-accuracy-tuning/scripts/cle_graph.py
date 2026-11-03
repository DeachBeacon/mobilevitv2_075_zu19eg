"""Export BN-folded, CLE or bounded-CLE FP32 segments and test sampled equivalence."""
from __future__ import print_function
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from tf_helpers import read_graph,import_graph,cpu_config,tf1
from cle_core import fold_front,feature_groups,equalize,apply_scales

def transform(source,input_tensor,output_tensor,samples_dir,out,variant='bounded',cap=32.,bias_cap=8.,activation_cap=8.,atol=1e-3,rtol=1e-5):
    if cap<1 or bias_cap<1 or activation_cap<1:raise ValueError('Scale/range caps must be >= 1')
    out=Path(os.path.realpath(str(out)))
    if out.exists():raise ValueError('Use a fresh output directory')
    source=Path(source);gd,original=read_graph(source)
    if original.get_tensor_by_name(input_tensor).dtype.name!='float32' or original.get_tensor_by_name(output_tensor).dtype.name!='float32':
        raise ValueError('This transform helper requires FP32 interfaces')
    if any(op.inputs[1].dtype.name!='float32' for op in original.get_operations() if op.type in ('Conv2D','DepthwiseConv2dNative')):
        raise ValueError('This transform helper requires FP32 convolution weights')
    if [op.outputs[0].name for op in original.get_operations() if op.type=='Placeholder']!=[input_tensor]:
        raise ValueError('Transform helper is single-input; partition a suitable CNN segment first')
    samples=sorted(Path(samples_dir).glob('*.npz'))
    if not samples:raise ValueError('Need numeric sample NPZ files keyed by the input tensor name')
    folded,layers=fold_front(gd,output_tensor)
    if any(n.op in ('FusedBatchNorm','FusedBatchNormV2','FusedBatchNormV3') for n in folded.node):
        raise ValueError('Remaining BN is not a direct supported Conv->BN pattern; adapt and verify the affine chain first')
    if not layers:raise ValueError('No Conv2D/DepthwiseConv2dNative layers found')
    stats=[];scale_record={};pinned=[]
    if variant=='bn':candidate=folded
    else:
        groups,members,pinned,_=feature_groups(folded,layers,input_tensor,output_tensor)
        full,history=equalize(layers,groups,members,pinned)
        scales=full
        if variant=='bounded':
            probes=[(group,name+':0') for group,names in members.items() if group not in pinned for name in names]
            ranges={g:np.zeros(groups.channels[g]) for g in members if g not in pinned}
            with tf1.Session(graph=import_graph(folded),config=cpu_config()) as s:
                for path in samples:
                    with np.load(str(path),allow_pickle=False) as data:image=data[input_tensor]
                    for (group,_),value in zip(probes,s.run([name for _,name in probes],{input_tensor:image})):
                        ranges[group]=np.maximum(ranges[group],np.abs(value).reshape(-1,value.shape[-1]).max(axis=0))
            scales={g:np.ones_like(v) for g,v in full.items()}
            for group,scale in full.items():
                if group in pinned:continue
                upper=np.full(scale.shape,cap)
                for layer in layers:
                    if layer['output_group']==group:
                        bias=np.abs(layer['bias']);upper=np.minimum(upper,bias_cap*max(float(bias.max()),1e-8)/np.maximum(bias,1e-12))
                activation=ranges[group]
                upper=np.minimum(upper,activation_cap*max(float(activation.max()),1e-8)/np.maximum(activation,1e-12))
                scales[group]=np.clip(scale,1./cap,upper)
        candidate,stats=apply_scales(folded,layers,scales)
        scale_record={g:v.tolist() for g,v in scales.items()}
    check={'samples':0,'max_abs_error':0.,'max_relative_l2':0.,'passed':True,'atol':atol,'relative_l2_limit':rtol}
    with tf1.Session(graph=original,config=cpu_config()) as ref,tf1.Session(graph=import_graph(candidate),config=cpu_config()) as actual:
        for path in samples:
            with np.load(str(path),allow_pickle=False) as data:value=data[input_tensor]
            if value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Finite FP32 calibration input required')
            expected=ref.run(output_tensor,{input_tensor:value});found=actual.run(output_tensor,{input_tensor:value})
            absolute=float(np.abs(found-expected).max());relative=float(np.linalg.norm(found-expected)/max(np.linalg.norm(expected),1e-12))
            check['samples']+=1;check['max_abs_error']=max(check['max_abs_error'],absolute)
            check['max_relative_l2']=max(check['max_relative_l2'],relative)
            check['passed'] &= bool(np.isfinite(found).all() and absolute<=atol and relative<=rtol)
    if not check['passed']:raise ValueError('FP32 equivalence failed: '+str(check))
    out.mkdir(parents=True);path=out/'float_model.pb';path.write_bytes(candidate.SerializeToString())
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report={'source_sha256':digest(source),'candidate_sha256':digest(path),'variant':variant,
        'input_tensor':input_tensor,'output_tensor':output_tensor,'verification':check,'pinned_groups':sorted(pinned),
        'scale_cap':cap if variant=='bounded' else None,'bias_cap_multiplier':bias_cap,'activation_cap_multiplier':activation_cap,
        'activation_statistic':'Maximum absolute value per channel across group members and supplied samples',
        'parameter_stats':stats,'scales':scale_record,'sample_hashes':{p.name:digest(p) for p in samples},
        'scope':'Host FP32 transform only; INT8 improvement, ELF/SO and board validation remain separate'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--variant',choices=('bn','cle','bounded'),default='bounded')
    p.add_argument('--scale-cap',type=float,default=32);p.add_argument('--bias-cap',type=float,default=8)
    p.add_argument('--activation-cap',type=float,default=8);p.add_argument('--atol',type=float,default=1e-3);p.add_argument('--relative-l2',type=float,default=1e-5)
    a=p.parse_args();print(transform(a.source,a.input,a.output,a.samples,a.out,a.variant,a.scale_cap,a.bias_cap,a.activation_cap,a.atol,a.relative_l2)['verification'])
