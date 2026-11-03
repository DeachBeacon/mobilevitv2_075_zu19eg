"""Rank measured classification reports only when their deployment scope matches."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rank(config_path,output):
    config_path=Path(config_path);cfg=read(config_path);reports=[]
    require(not Path(output).exists(),'Use a new selection file')
    for file in cfg['reports']:
        path=config_path.parent/file;r=read(path);rows=r['rows']
        require(rows and len({row['sample_id'] for row in rows})==len(rows),'Empty/duplicate sample rows')
        require(all(row['label'] is not None and row['prediction'] is not None for row in rows),'Known labels and predictions required')
        correct=sum(row['label']==row['prediction'] for row in rows)
        require(correct==r['correct_images'] and r['sample_count']==len(rows) and
                abs(r['accuracy']-correct/float(len(rows)))<1e-12,'Accuracy summary inconsistent with rows')
        require(r['evaluation_mode']==cfg['evaluation_mode'] and r['backend']==cfg['backend'],'Wrong backend or evaluation scope')
        require(r['evidence']=='measured','Host fixture/simulation results cannot stand in for native or board measurements')
        require(all(re.match(r'^[0-9a-f]{64}$',r[key]) for key in ('dataset_sha256','preprocess_sha256','fixed_artifacts_sha256')),
                'Dataset/preprocessing/frozen-backend fingerprints must be actual SHA256 values')
        require(r['artifacts'],'Need evaluated artifact paths and hashes')
        for filename,h in r['artifacts'].items():require(sha(path.parent/filename)==h,'Evaluated artifact changed: '+filename)
        r=dict(r,report_path=str(path),report_sha256=sha(path));reports.append(r)
    require(len({r['candidate_id'] for r in reports})==len(reports),'Duplicate candidate IDs')
    baseline=next((r for r in reports if r['candidate_id']==cfg['baseline_id']),None)
    require(baseline is not None and baseline['deployable'] is True,'Need a deployable baseline measured in this same scope')
    signature=('dataset_sha256','preprocess_sha256','fixed_artifacts_sha256','evaluation_mode','backend','input_range_policy')
    sample_signature=lambda r:sorted((row['sample_id'],str(row['label'])) for row in r['rows'])
    for r in reports:
        require(all(r[k]==baseline[k] for k in signature),'Incomparable dataset, preprocessing, backend, fixed models or range policy')
        require(sample_signature(r)==sample_signature(baseline),'Candidate uses a different sample/label set')
    safe=[r for r in reports if r['deployable'] is True]
    best=max(safe,key=lambda r:(r['correct_images'],r['candidate_id']==cfg['baseline_id']))
    portable={os.path.relpath(str(Path(best['report_path']).parent/path),str(Path(output).parent)):h for path,h in best['artifacts'].items()}
    result={'candidate_id':best['candidate_id'],'accuracy':best['accuracy'],'correct_images':best['correct_images'],
            'gain_images':best['correct_images']-baseline['correct_images'],'evaluation_mode':cfg['evaluation_mode'],
            'backend':cfg['backend'],'report_sha256':best['report_sha256'],'artifacts':portable,
            'report_path':os.path.relpath(best['report_path'],str(Path(output).parent)),
            'decision':'keep_control' if best['candidate_id']==cfg['baseline_id'] else 'candidate',
            'scope':'Comparable reported classification results; this arithmetic/provenance check does not execute or authenticate hardware'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(rank(a.config,a.out),indent=2))
