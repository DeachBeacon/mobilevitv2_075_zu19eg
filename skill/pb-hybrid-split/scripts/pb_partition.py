"""Inspect, partition and verify a numeric stateless frozen TensorFlow GraphDef.

PC helper: TensorFlow 1.15 + NumPy; no compiler or board runtime is invoked.
"""
from __future__ import print_function
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import tensorflow as tf
T=tf.compat.v1

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def absolute(path):return Path(os.path.realpath(str(path)))
def require(ok,message):
    if not ok:raise ValueError(message)
def tensor_name(name):return name if ':' in name else name+':0'
def node_name(edge):return edge.lstrip('^').split(':')[0]
def graph_of(gd):
    graph=tf.Graph()
    with graph.as_default():T.import_graph_def(gd,name='')
    return graph
def read_model(path):
    gd=T.GraphDef();gd.ParseFromString(Path(path).read_bytes())
    require(not any(n.op=='FixNeuron' or 'wpos' in n.attr for n in gd.node),
            'Use the original FP32 GraphDef, not a quantized eval/deploy PB')
    graph=graph_of(gd)
    stateful=[op.name for op in graph.get_operations() if op.op_def.is_stateful]
    require(not stateful,'Freeze/remove stateful operations before partitioning: '+str(stateful[:12]))
    return gd,graph
def description(t):return {'tensor':t.name,'dtype':t.dtype.name,'shape':t.shape.as_list() if t.shape.ndims is not None else None}
def config():return T.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)

def inspect(path):
    gd,g=read_model(path)
    return {'source_sha256':sha(path),'nodes':len(gd.node),'ops':dict(Counter(n.op for n in gd.node)),
            'inputs':[description(op.outputs[0]) for op in g.get_operations() if op.type=='Placeholder'],
            'sinks':[description(t) for op in g.get_operations() for t in op.outputs if not t.consumers()],
            'scope':'Static inventory, not a DPU support verdict'}

def export_stage(gd,original,part):
    require(re.match(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$',part['name']), 'Unsafe or invalid part name')
    mapped={};bindings=[];aliases=set();modified=T.GraphDef();modified.CopyFrom(gd)
    existing={n.name:n for n in modified.node}
    for index,item in enumerate(part['inputs']):
        src=tensor_name(item['tensor']);tensor=original.get_tensor_by_name(src)
        alias=item.get('placeholder','boundary_'+str(index))
        require(src not in mapped and alias not in aliases,'Duplicate stage input or alias')
        require(np.dtype(tensor.dtype.as_numpy_dtype).kind in 'biuf','Real numeric input required; encoded/resource/complex input needs a dedicated adapter')
        shape=item['shape'] if 'shape' in item else tensor.shape.as_list()
        require(tf.TensorShape(shape).is_compatible_with(tensor.shape),'Incompatible input shape override: '+src)
        if alias in existing:
            require(src==alias+':0' and existing[alias].op=='Placeholder','Alias collides with a source operation: '+alias)
            n=existing[alias];n.attr['shape'].shape.CopyFrom(tf.TensorShape(shape).as_proto())
        else:
            n=modified.node.add();n.name=alias;n.op='Placeholder'
            n.attr['dtype'].type=tensor.dtype.as_datatype_enum;n.attr['shape'].shape.CopyFrom(tf.TensorShape(shape).as_proto())
        mapped[src]=alias+':0';aliases.add(alias)
        bindings.append({'source_tensor':src,'local_tensor':alias+':0','dtype':tensor.dtype.name,'shape':shape})
    # Rewrite data edges by tensor PORT. Control edges are retained, never guessed.
    for n in modified.node:
        for i,edge in enumerate(n.input):
            if not edge.startswith('^') and tensor_name(edge) in mapped:n.input[i]=mapped[tensor_name(edge)]
        n.device=''
    outputs=[]
    for name in part['outputs']:
        src=tensor_name(name);original.get_tensor_by_name(src)
        outputs.append({'source_tensor':src,'local_tensor':mapped.get(src,src)})
    sliced=T.graph_util.extract_sub_graph(modified,[node_name(o['local_tensor']) for o in outputs])
    exported=graph_of(sliced)
    found={op.name for op in exported.get_operations() if op.type=='Placeholder'}
    require(found==aliases,'Incomplete boundary for '+part['name']+': live inputs '+str(sorted(found))+
            ', declared '+str(sorted(aliases))+'. Include skip/residual/control dependencies explicitly.')
    for o in outputs:o.update(description(exported.get_tensor_by_name(o['local_tensor'])))
    if part.get('require_static_nhwc4'):
        for item in bindings+outputs:
            require(item['shape'] is not None and len(item['shape'])==4 and all(v is not None and v>0 for v in item['shape']),
                    'This target requires static rank-4 interfaces: '+str(item))
    return sliced,{'name':part['name'],'device':part['device'],'inputs':bindings,'outputs':outputs,
                   'ops':dict(Counter(n.op for n in sliced.node))}

def split(source,plan_path,out):
    source=absolute(source);out=absolute(out);plan=load_json(plan_path);gd,graph=read_model(source)
    require(not out.exists(),'Use a fresh output directory; original artifacts are preserved')
    names=[p['name'] for p in plan['parts']];require(len(names)==len(set(names)),'Duplicate part names')
    prepared=[]
    for p in plan['parts']:prepared.append(export_stage(gd,graph,p))
    available={tensor_name(n) for n in plan['model_inputs']}
    for _,p in prepared:
        require(all(i['source_tensor'] in available for i in p['inputs']),'Stage order or missing producer: '+p['name'])
        available.update(o['source_tensor'] for o in p['outputs'])
    require(all(tensor_name(n) in available for n in plan['model_outputs']),'Final outputs are not produced')
    actual={op.outputs[0].name for op in graph.get_operations() if op.type=='Placeholder'}
    require(actual=={tensor_name(n) for n in plan['model_inputs']},'Declare every original input exactly')
    out.mkdir(parents=True);parts=[]
    for gd,p in prepared:
        path=out/(p['name']+'.pb');path.write_bytes(gd.SerializeToString());p.update(file=path.name,sha256=sha(path));parts.append(p)
    manifest={'format_version':1,'source_sha256':sha(source),'plan_sha256':sha(plan_path),
              'model_inputs':[description(graph.get_tensor_by_name(tensor_name(n))) for n in plan['model_inputs']],
              'model_outputs':[tensor_name(n) for n in plan['model_outputs']], 'parts':parts,
              'scope':'FP32 partition only; compilation, INT8 accuracy and board performance are not established'}
    save_json(out/'manifest.json',manifest);return manifest

def compare(actual,reference,atol,rtol):
    good=actual.shape==reference.shape and actual.dtype==reference.dtype
    finite=bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not good or not finite:return {'passed':False,'shape_dtype_match':good,'finite':finite}
    delta=actual.astype(np.float64)-reference.astype(np.float64)
    passed=np.array_equal(actual,reference) if actual.dtype.kind in 'biu' else np.allclose(actual,reference,atol=atol,rtol=rtol)
    return {'passed':bool(passed),
            'max_abs_error':float(np.abs(delta).max()),
            'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(reference.astype(np.float64)),1e-12))}

def verify(source,parts_dir,samples_dir,report,atol=1e-5,rtol=1e-5,numpy_dir=None):
    parts_dir=absolute(parts_dir);m=load_json(parts_dir/'manifest.json');gd,graph=read_model(source)
    require(sha(source)==m['source_sha256'],'Golden source PB changed')
    require(not Path(report).exists(),'Verification report exists; choose a new report path')
    samples=sorted(Path(samples_dir).glob('*.npz'));require(samples,'Need NPZ files containing original numeric model inputs')
    sessions=[];numpy_models={};results=[]
    full=T.Session(graph=graph,config=config())
    try:
        for part in m['parts']:
            path=parts_dir/part['file'];require(sha(path)==part['sha256'],'Part PB changed: '+part['name'])
            sessions.append(T.Session(graph=graph_of(read_model(path)[0]),config=config()))
            if numpy_dir and (Path(numpy_dir)/part['name']/'graph.json').exists():
                from numpy_runtime import NumpySegment
                folder=Path(numpy_dir)/part['name'];meta=load_json(folder/'manifest.json')
                require(meta['source_pb_sha256']==sha(path),'NumPy export comes from a different PB')
                for filename,h in meta['files'].items():require(sha(folder/filename)==h,'NumPy export changed')
                numpy_models[part['name']]=NumpySegment(folder/'graph.json')
        fetches=sorted(set(m['model_outputs']+[o['source_tensor'] for p in m['parts'] for o in p['outputs']]))
        for file in samples:
            with np.load(str(file),allow_pickle=False) as data:
                values={i['tensor']:data[i['tensor']] for i in m['model_inputs']}
            for i in m['model_inputs']:
                require(values[i['tensor']].dtype==np.dtype(i['dtype']),'Input dtype mismatch: '+i['tensor'])
                require(np.isfinite(values[i['tensor']]).all(),'Non-finite numeric input')
            golden=dict(zip(fetches,full.run(fetches,values)));checks=[]
            for p,s in zip(m['parts'],sessions):
                feed={i['local_tensor']:values[i['source_tensor']] for i in p['inputs']}
                output=s.run([o['local_tensor'] for o in p['outputs']],feed)
                if p['name'] in numpy_models:
                    require(len(feed)==len(output)==1,'Optional NumPy adapter is single input/output')
                    np_output=numpy_models[p['name']].predict(next(iter(feed.values())))
                    checks.append(dict(compare(np_output,output[0],atol,rtol),part=p['name'],backend='numpy'))
                for o,v in zip(p['outputs'],output):
                    checks.append(dict(compare(v,golden[o['source_tensor']],atol,rtol),part=p['name'],tensor=o['source_tensor']))
                    values[o['source_tensor']]=v
            results.append({'sample':file.name,'passed':all(c['passed'] for c in checks),'checks':checks})
    finally:
        full.close()
        for s in sessions:s.close()
    result={'passed':all(r['passed'] for r in results),'samples':len(results),'source_sha256':sha(source),
            'atol':atol,'rtol':rtol,'scope':'Host saved-PB chain and optional NumPy parity only','results':results}
    save_json(report,result);require(result['passed'],'Numerical parity failed; see report');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('inspect','split','verify'))
    p.add_argument('--source',type=Path,required=True);p.add_argument('--plan',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--parts',type=Path);p.add_argument('--samples',type=Path);p.add_argument('--report',type=Path)
    p.add_argument('--numpy-dir',type=Path);p.add_argument('--atol',type=float,default=1e-5);p.add_argument('--rtol',type=float,default=1e-5)
    a=p.parse_args()
    if a.action=='inspect':
        value=inspect(a.source)
        if a.report:save_json(a.report,value)
        print(json.dumps(value,indent=2))
    elif a.action=='split':
        if not a.plan or not a.out:p.error('split requires --plan and --out')
        print(json.dumps(split(a.source,a.plan,a.out),indent=2))
    else:
        if not a.parts or not a.samples or not a.report:p.error('verify requires --parts, --samples and --report')
        print(verify(a.source,a.parts,a.samples,a.report,a.atol,a.rtol,a.numpy_dir)['passed'])
