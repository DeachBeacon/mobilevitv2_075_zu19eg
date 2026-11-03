"""Audit/copy-patch DNNDK/Vitis-AI-1.x input quantization tags; never edit SOs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from tensorflow.core.framework.graph_pb2 import GraphDef

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_pb(path):
    graph=GraphDef();graph.ParseFromString(Path(path).read_bytes());return graph
def trace(graph,name):
    todo=[name];seen={name};bridges=[];convs=[]
    while todo:
        parent=todo.pop()
        for n in graph.node:
            if not n.input or n.input[0].startswith('^') or n.input[0].split(':')[0]!=parent:continue
            if n.op in ('Conv2D','DepthwiseConv2dNative'):convs.append(n);continue
            require(n.op in ('FixNeuron','Pad','Identity'),'Unsupported boundary operation: '+n.op)
            if n.name not in seen:seen.add(n.name);bridges.append(n);todo.append(n.name)
    require(convs,'No first convolution found');return bridges,convs
def slots(evaluation,deployment,input_tensor):
    name=input_tensor.split(':')[0]
    require(input_tensor==name+':0','Input placeholder must use port 0')
    e,_=trace(evaluation,name);d,convs=trace(deployment,name)
    fixes=[n for n in e if n.op=='FixNeuron']
    placeholders=[n for n in deployment.node if n.name==name and n.op=='Placeholder']
    require(fixes and len(placeholders)==1,'Missing input FixNeuron or deploy Placeholder')
    scalar=[(n,'quantize_pos') for n in fixes]
    vectors=[(placeholders[0],'opos')]+[(n,'ipos') for n in convs]
    for n in d:
        require(n.op in ('Pad','Identity'),'Unexpected deploy boundary operation')
        if n.op=='Pad':vectors.extend([(n,'ipos'),(n,'opos')])
        else:
            vectors.extend((n,key) for key in ('ipos','opos') if key in n.attr)
    require(all(n.attr['bit_width'].i==8 for n,_ in scalar),'This legacy adapter expects 8-bit FixNeuron')
    require(all(len(n.attr[key].list.i)==2 and n.attr[key].list.i[0]==8 for n,key in vectors),'Missing/unsupported deploy input tags')
    return scalar,vectors
def audit(evaluation,deployment,input_tensor):
    scalar,vectors=slots(evaluation,deployment,input_tensor)
    rows=[{'node':n.name,'attribute':key,'position':int(n.attr[key].i)} for n,key in scalar]
    rows.extend({'node':n.name,'attribute':key,'position':int(n.attr[key].list.i[1])} for n,key in vectors)
    positions={r['position'] for r in rows};consistent=len(positions)==1
    report={'consistent':consistent,'tags':rows,'positions':sorted(positions),
            'scope':'Legacy 8-bit single input boundary metadata, not internal quantization or accuracy validation'}
    if consistent:
        position=next(iter(positions));scale=2.**position
        report.update(input_scale=scale,representable_interval=[-128./scale,127./scale])
    return report
def run(eval_pb,deploy_pb,input_tensor,position=None,out=None,samples=None):
    e,d=read_pb(eval_pb),read_pb(deploy_pb);report=audit(e,d,input_tensor)
    report['source_hashes']={'eval':sha(eval_pb),'deploy':sha(deploy_pb)}
    if position is not None:
        require(out is not None,'--out is required for --apply-position')
        out=Path(os.path.realpath(str(out)));require(not out.exists(),'Use a fresh output directory')
        scalar,vectors=slots(e,d,input_tensor)
        for n,key in scalar:n.attr[key].i=int(position)
        for n,key in vectors:n.attr[key].list.i[1]=int(position)
        out.mkdir(parents=True)
        for name,g in [('quantize_eval_model.pb',e),('deploy_model.pb',d)]:
            (out/name).write_bytes(g.SerializeToString())
        report={'before':report,'after':audit(e,d,input_tensor),
                'files':{name:sha(out/name) for name in ('quantize_eval_model.pb','deploy_model.pb')},
                'invalidated':'Reevaluate native accuracy and downstream calibration, then recompile/relink. Old ELF/SO cannot be reused for changed PBs.'}
    active=report.get('after',report)
    if samples:
        require(active['consistent'],'Resolve inconsistent metadata before range audit')
        low,high=active['representable_interval'];outside=0;count=0
        for path in sorted(Path(samples).glob('*.npz')):
            with np.load(str(path),allow_pickle=False) as data:value=data[input_tensor]
            require(value.dtype==np.float32 and np.isfinite(value).all(),'Need finite FP32 boundary features')
            outside+=int(np.count_nonzero((value<low)|(value>high)));count+=int(value.size)
        require(count>0,'No boundary sample elements found')
        report['range_audit']={'elements':count,'outside_elements':outside,'scope':'Conservative endpoints; runtime rounding is not emulated'}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--eval-pb',type=Path,required=True)
    p.add_argument('--deploy-pb',type=Path,required=True);p.add_argument('--input',required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--expect-position',type=int)
    p.add_argument('--apply-position',type=int);p.add_argument('--out',type=Path);p.add_argument('--samples',type=Path)
    a=p.parse_args();require(not a.report.exists(),'Choose a new report path')
    result=run(a.eval_pb,a.deploy_pb,a.input,a.apply_position,a.out,a.samples)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    active=result.get('after',result);print(json.dumps(result,indent=2))
    if not active['consistent'] or (a.expect_position is not None and active['positions']!=[a.expect_position]):raise SystemExit(1)
    if result.get('range_audit',{}).get('outside_elements',0):raise SystemExit(1)
