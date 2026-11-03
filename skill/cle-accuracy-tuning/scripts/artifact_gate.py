"""Legacy AArch64 DPU artifact consistency gate; standard library, Python 3.5+."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def record(path,value):
    path=Path(path)
    if path.exists():require(read(path)==value,'Recorded artifacts changed; use a new build directory')
    else:path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def check_selection(path,deploy):
    path=Path(path);s=read(path)
    require(s['backend'] in ('native_fixneuron','board'),'Compilation needs native or board scoring, not FP32-only tests')
    require(sha(path.parent/s['report_path'])==s['report_sha256'],'Selected measurement report changed')
    for file,h in s['artifacts'].items():require(sha(path.parent/file)==h,'Evaluated artifact changed: '+file)
    require(sha(deploy) in s['artifacts'].values(),'Deploy PB was not included in selected evaluated artifacts')
    return s
def sections(path,kind,kernel):
    raw=Path(path).read_bytes()
    require(len(raw)>=64 and raw[:6]==b'\x7fELF\x02\x01','Expected little-endian ELF64')
    require(struct.unpack_from('<HH',raw,16)==(kind,183),'This adapter requires AArch64 ELF type '+str(kind))
    offset=struct.unpack_from('<Q',raw,40)[0];size,count,strings=struct.unpack_from('<HHH',raw,58)
    require(size>=64 and strings<count and offset+size*count<=len(raw),'Invalid section table')
    entries=[struct.unpack_from('<IIQQQQIIQQ',raw,offset+i*size) for i in range(count)]
    table=entries[strings];require(table[4]+table[5]<=len(raw),'Invalid section names')
    names=raw[table[4]:table[4]+table[5]];result={}
    for entry in entries:
        name=names[entry[0]:].split(b'\0',1)[0].decode('ascii')
        if name.startswith('.deephi.'):
            require(entry[4]+entry[5]<=len(raw),'Truncated DPU section')
            result[name]=hashlib.sha256(raw[entry[4]:entry[4]+entry[5]]).hexdigest()
    require('.deephi.metadata.'+kernel in result,'Kernel name does not match embedded DPU metadata')
    return result
def compiled(build,kernel,selection,deploy):
    build=Path(build);check_selection(selection,deploy);folder=build/'compile'
    gv=folder/(kernel+'_kernel_graph.gv');text=gv.read_text(encoding='utf-8')
    require(text.count('type: DPUKernel')==1 and 'CPUKernel' not in text,'This CNN adapter expects one DPU kernel with no CPU fallback')
    elf=folder/('dpu_'+kernel+'.elf');regions=sections(elf,1,kernel)
    log=(folder/'compile.log').read_text(encoding='utf-8')
    require('Input Node(s)' in log and 'Output Node(s)' in log,'Compiler runtime boundary log missing')
    nodes=[list(item) for item in re.findall(r'([\w./-]+)\(0\)\s*:\s*([0-9 *]+)',log)]
    value={'kernel':kernel,'selection_sha256':sha(selection),'deploy_sha256':sha(deploy),'elf_sha256':sha(elf),
           'graph_sha256':sha(gv),'compile_log_sha256':sha(folder/'compile.log'),
           'runtime_nodes':nodes,'dpu_sections':regions}
    return value

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('float','selection','compiled','linked'))
    p.add_argument('--model',type=Path);p.add_argument('--float-manifest',type=Path)
    p.add_argument('--selection',type=Path);p.add_argument('--deploy',type=Path);p.add_argument('--build',type=Path);p.add_argument('--kernel')
    a=p.parse_args()
    if a.action=='float':
        m=read(a.float_manifest);require(m['verification']['passed'] and m['candidate_sha256']==sha(a.model),'Missing or stale FP32 transform verification')
    elif a.action=='selection':check_selection(a.selection,a.deploy)
    else:
        value=compiled(a.build,a.kernel,a.selection,a.deploy);path=a.build/'compile/compile_manifest.json'
        if a.action=='compiled':record(path,value)
        else:
            require(read(path)==value,'ELF/selection/compile log changed')
            so=a.build/'compile'/('libdpumodel'+a.kernel+'.so')
            require(sections(so,3,a.kernel)==value['dpu_sections'],'Linked SO DPU payload differs from selected ELF')
            record(a.build/'compile/linked_manifest.json',dict(value,so_sha256=sha(so)))
    print('Artifact consistency check passed:',a.action)
