"""Export a verified single-input/output FP32 segment to a finite NumPy backend."""
import argparse
from pathlib import Path
import numpy as np
from pb_partition import read_model,sha,save_json,require,absolute
from numpy_export_core import prepare_numpy_export

def export(pb,input_tensor,output_tensor,out):
    out=absolute(out);require(not out.exists(),'Use a fresh NumPy output directory')
    gd,g=read_model(pb);inp=g.get_tensor_by_name(input_tensor);output=g.get_tensor_by_name(output_tensor)
    placeholders=[op.outputs[0].name for op in g.get_operations() if op.type=='Placeholder']
    require(placeholders==[input_tensor],'NumPy adapter supports exactly one declared input')
    require(inp.dtype.name==output.dtype.name=='float32','NumPy adapter requires FP32 input/output')
    for tensor in (inp,output):require(all(v is not None and v>0 for v in tensor.shape.as_list()),'Use static interfaces')
    for op in g.get_operations():
        if op.type in ('Conv2D','DepthwiseConv2dNative'):
            require(op.get_attr('padding') in (b'SAME',b'VALID'),'EXPLICIT padding needs an additional tested adapter')
    record,weights=prepare_numpy_export(gd,input_tensor,output_tensor)
    record.update(input_shape=inp.shape.as_list(),output_shape=output.shape.as_list(),weights_file='weights.npz')
    out.mkdir(parents=True);save_json(out/'graph.json',record);np.savez_compressed(str(out/'weights.npz'),**weights)
    save_json(out/'manifest.json',{'source_pb_sha256':sha(pb),
        'files':{n:sha(out/n) for n in ('graph.json','weights.npz')},
        'scope':'Finite operator adapter export; use pb_partition verify --numpy-dir before acceptance'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pb',type=Path,required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();export(a.pb,a.input,a.output,a.out)
