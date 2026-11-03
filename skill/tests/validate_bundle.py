"""Portable host regression tests. Optional --real-case-root adds the retained case.

No Docker, ARM execution, native FixNeuron inference or PYNQ hardware is emulated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[1]
SPLIT=ROOT/'pb-hybrid-split/scripts';CLE=ROOT/'cle-accuracy-tuning/scripts'
sys.path[:0]=[str(SPLIT),str(CLE)]
import pb_partition as pp
import export_numpy as ne
from numpy_runtime import NumpySegment
import cle_graph as cg
import quant_boundary as qb
import rank_candidates as rc
import artifact_gate as ag
import tensor_contract as tc
T=tf.compat.v1

def save(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def make_model(folder,branched=False,cnn=False,sigmoid=False):
    g=tf.Graph();rng=np.random.RandomState(17)
    with g.as_default():
        x=T.placeholder(tf.float32,[1,4,4,3],name='image')
        if cnn:
            w=tf.constant(rng.normal(size=(1,1,3,4)).astype(np.float32)*np.array([.01,.1,2.,8.],np.float32))
            h=tf.nn.conv2d(x,w,[1,1,1,1],'SAME',name='conv1')
            h=tf.nn.fused_batch_norm(h,tf.constant([1.2,-.6,.9,2.]),tf.constant([.1,.2,-.1,.3]),
                mean=tf.constant([.2,.1,-.2,.4]),variance=tf.constant([1.,2.,.5,3.]),is_training=False,name='bn1')[0]
            h=tf.nn.relu(h,name='hidden1')
            dw=tf.constant(rng.normal(size=(3,3,4,1)).astype(np.float32)*.1)
            h=tf.nn.depthwise_conv2d(h,dw,[1,1,1,1],'SAME',name='depthwise')
            h=tf.nn.bias_add(h,tf.constant([.1,-.2,.3,.4]),name='dw_bias')
            h=tf.nn.relu(h,name='hidden2')
            h=tf.nn.conv2d(h,tf.constant(rng.normal(size=(1,1,4,3)).astype(np.float32)*.1),[1,1,1,1],'SAME',name='projection')
            y=tf.add(h,x,name='result')
            if sigmoid:y=tf.sigmoid(y,name='sigmoid_result')
        else:
            h=tf.nn.conv2d(x,tf.constant(rng.normal(size=(1,1,3,4)).astype(np.float32)*.1),[1,1,1,1],'SAME',name='conv')
            h=tf.nn.relu(h,name='features')
            if branched:
                a,b=tf.split(h,[2,2],axis=3,name='fork')
                y=tf.add(a*3.+b,tf.reduce_mean(x,axis=3,keepdims=True),name='logits')
            else:
                y=tf.matmul(tf.reshape(h,[1,64]),tf.constant(rng.normal(size=(64,5)).astype(np.float32)*.01),name='logits')
    path=folder/'model.pb';path.write_bytes(g.as_graph_def().SerializeToString())
    samples=folder/'samples';samples.mkdir()
    for i in range(3):np.savez(str(samples/('sample'+str(i)+'.npz')),**{'image:0':rng.normal(size=(1,4,4,3)).astype(np.float32)})
    return path,samples,y.name

class SplitTests(unittest.TestCase):
    def test_explicit_shape_specializes_unknown_rank_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g=tf.Graph()
            with g.as_default():
                x=T.placeholder(tf.float32,shape=None,name='image')
                h=tf.multiply(x,2.,name='features');tf.add(h,1.,name='logits')
            source=root/'model.pb';source.write_bytes(g.as_graph_def().SerializeToString())
            plan={'model_inputs':['image:0'],'model_outputs':['logits:0'],'parts':[
                {'name':'front','device':'ps','inputs':[{'tensor':'image:0','placeholder':'image','shape':[1,2]}],'outputs':['features:0']},
                {'name':'tail','device':'ps','inputs':[{'tensor':'features:0','placeholder':'tail_input','shape':[1,2]}],'outputs':['logits:0']}]}
            save(root/'plan.json',plan);pp.split(source,root/'plan.json',root/'split')
            samples=root/'samples';samples.mkdir();np.savez(str(samples/'one.npz'),**{'image:0':np.array([[1.,2.]],np.float32)})
            self.assertTrue(pp.verify(source,root/'split',samples,root/'report.json')['passed'])

    def test_quantized_graph_is_not_accepted_as_float_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'eval.pb';g=T.GraphDef();node=g.node.add();node.name='fake_quantizer';node.op='FixNeuron'
            path.write_bytes(g.SerializeToString())
            with self.assertRaisesRegex(ValueError,'original FP32'):pp.read_model(path)

    def test_multiport_multiinput_and_skip_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,samples,output=make_model(root,branched=True)
            plan={'model_inputs':['image:0'],'model_outputs':[output],'parts':[
                {'name':'front','device':'dpu','inputs':[{'tensor':'image:0','placeholder':'image'}],'outputs':['fork:1','fork:0']},
                {'name':'tail','device':'ps','inputs':[{'tensor':'fork:0','placeholder':'a'},
                    {'tensor':'fork:1','placeholder':'b'},{'tensor':'image:0','placeholder':'shortcut'}],'outputs':[output]}]}
            file=root/'plan.json';save(file,plan);original=pp.sha(source)
            m=pp.split(source,file,root/'split')
            result=pp.verify(source,root/'split',samples,root/'verify.json')
            self.assertTrue(result['passed']);self.assertEqual(result['samples'],3)
            self.assertEqual(m['parts'][0]['outputs'][0]['source_tensor'],'fork:1')
            self.assertEqual(pp.sha(source),original)
            plan['parts'][1]['inputs'].pop();save(file,plan)
            with self.assertRaisesRegex(ValueError,'Incomplete boundary'):pp.split(source,file,root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_optional_numpy_and_no_tensorflow_runtime_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,samples,output=make_model(root)
            plan={'model_inputs':['image:0'],'model_outputs':[output],'parts':[
                {'name':'front','device':'dpu','inputs':[{'tensor':'image:0','placeholder':'image'}],'outputs':['features:0']},
                {'name':'tail','device':'ps','inputs':[{'tensor':'features:0','placeholder':'ps_features'}],'outputs':[output]}]}
            save(root/'plan.json',plan);pp.split(source,root/'plan.json',root/'split')
            ne.export(root/'split/tail.pb','ps_features:0',output,root/'numpy/tail')
            self.assertTrue(pp.verify(source,root/'split',samples,root/'verify.json',numpy_dir=root/'numpy')['passed'])
            data=np.zeros((1,4,4,4),np.float32);np.save(str(root/'input.npy'),data,allow_pickle=False)
            code='''import builtins,sys
old=builtins.__import__
def blocked(name,*a,**kw):
    if name=="tensorflow" or name.startswith("tensorflow."):raise ImportError("TF blocked")
    return old(name,*a,**kw)
builtins.__import__=blocked
sys.path.insert(0,sys.argv[1])
from numpy_runtime import NumpySegment
import numpy as np
value=NumpySegment(sys.argv[2]).predict(np.load(sys.argv[3],allow_pickle=False))
assert value.shape==(1,5) and np.isfinite(value).all()
assert not any(n=="tensorflow" or n.startswith("tensorflow.") for n in sys.modules)
'''
            child=subprocess.run([sys.executable,'-B','-c',code,str(SPLIT),str(root/'numpy/tail/graph.json'),str(root/'input.npy')],
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
            self.assertEqual(child.returncode,0,child.stderr)

    def test_saved_artifact_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,samples,output=make_model(root)
            plan={'model_inputs':['image:0'],'model_outputs':[output],'parts':[
                {'name':'all','device':'ps','inputs':[{'tensor':'image:0','placeholder':'image'}],'outputs':[output]}]}
            save(root/'plan.json',plan);pp.split(source,root/'plan.json',root/'split')
            with (root/'split/all.pb').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Part PB changed'):pp.verify(source,root/'split',samples,root/'report.json')

    def test_numpy_unknown_operator_fails_instead_of_being_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,_,output=make_model(root,cnn=True,sigmoid=True)
            with self.assertRaisesRegex(ValueError,'does not implement'):ne.export(source,'image:0',output,root/'numpy')

    def test_generic_boundary_range_and_noncontiguous_transfer(self):
        value=np.array([-128.,127.,-129.],np.float32)
        contract={'shape':[3],'host_dtype':'float32','device_dtype':'int8','input_scale':1.}
        before=value.copy();report=tc.audit(value,contract)
        self.assertFalse(report['passed']);self.assertEqual(report['outside_elements'],1)
        np.testing.assert_array_equal(value,before)
        self.assertFalse(tc.audit(np.zeros(3,np.float32),{'shape':[3],'host_dtype':'float32','device_dtype':'int8'})['passed'])
        v=np.arange(9,dtype=np.float32).reshape(3,3).T
        self.assertFalse(tc.audit(v,{'shape':[3,3],'host_dtype':'float32'})['passed'])

class CleTests(unittest.TestCase):
    def test_self_coupled_internal_residual_group_is_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g=tf.Graph()
            with g.as_default():
                x=T.placeholder(tf.float32,[1,2,2,2],name='image')
                w=tf.constant(np.array([[[[.2,.1],[.3,.4]]]],np.float32))
                h=tf.nn.conv2d(x,w,[1,1,1,1],'SAME',name='before')
                loop=tf.nn.conv2d(h,w,[1,1,1,1],'SAME',name='inside')
                h=tf.nn.relu(tf.add(h,loop,name='residual'),name='hidden')
                y=tf.nn.conv2d(h,w,[1,1,1,1],'SAME',name='after')
            path=root/'model.pb';path.write_bytes(g.as_graph_def().SerializeToString())
            samples=root/'samples';samples.mkdir();np.savez(str(samples/'one.npz'),**{'image:0':np.ones((1,2,2,2),np.float32)})
            report=cg.transform(path,'image:0',y.name,samples,root/'cle','cle')
            self.assertTrue(report['verification']['passed'])
            self.assertGreaterEqual(len(report['pinned_groups']),3)
            self.assertTrue(all(all(v==1. for v in report['scales'][group]) for group in report['pinned_groups']))

    def test_bn_cle_bounded_depthwise_and_residual_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,samples,output=make_model(root,cnn=True);before=pp.sha(source)
            for variant in ('bn','cle','bounded'):
                report=cg.transform(source,'image:0',output,samples,root/variant,variant,cap=8,bias_cap=4,activation_cap=4)
                self.assertTrue(report['verification']['passed'])
                for group in report['pinned_groups']:np.testing.assert_array_equal(report['scales'][group],np.ones(len(report['scales'][group])))
                if variant=='bounded':self.assertTrue(all(max(s)<=8. for s in report['scales'].values()))
            self.assertEqual(pp.sha(source),before)

    def test_nonhomogeneous_activation_is_not_blindly_equalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,samples,output=make_model(root,cnn=True,sigmoid=True)
            with self.assertRaisesRegex(ValueError,'Unsupported feature operator'):
                cg.transform(source,'image:0',output,samples,root/'cle','cle')

    def test_legacy_input_pad_metadata_copy_patch_and_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e=T.GraphDef();d=T.GraphDef()
            def node(g,name,op,inputs=()):
                n=g.node.add();n.name=name;n.op=op;n.input.extend(inputs);return n
            node(e,'sensor','Placeholder');f=node(e,'sensor/aquant','FixNeuron',['sensor'])
            f.attr['bit_width'].i=8;f.attr['quantize_pos'].i=3
            node(e,'pad','Pad',['sensor/aquant']);f=node(e,'pad/aquant','FixNeuron',['pad'])
            f.attr['bit_width'].i=8;f.attr['quantize_pos'].i=3;node(e,'conv','Conv2D',['pad/aquant'])
            n=node(d,'sensor','Placeholder');n.attr['opos'].list.i.extend([8,3])
            n=node(d,'pad','Pad',['sensor']);n.attr['ipos'].list.i.extend([8,3]);n.attr['opos'].list.i.extend([8,3])
            n=node(d,'conv','Conv2D',['pad']);n.attr['ipos'].list.i.extend([8,4])
            ep=root/'eval.pb';dp=root/'deploy.pb';ep.write_bytes(e.SerializeToString());dp.write_bytes(d.SerializeToString())
            before=(pp.sha(ep),pp.sha(dp));self.assertFalse(qb.run(ep,dp,'sensor:0')['consistent'])
            samples=root/'samples';samples.mkdir();np.savez(str(samples/'one.npz'),**{'sensor:0':np.array([-33.,0.,31.75],np.float32)})
            report=qb.run(ep,dp,'sensor:0',2,root/'fixed',samples)
            self.assertTrue(report['after']['consistent']);self.assertEqual(report['after']['positions'],[2])
            self.assertEqual(report['range_audit']['outside_elements'],1)
            self.assertEqual((pp.sha(ep),pp.sha(dp)),before)

    def reports(self,root):
        fingerprint=lambda t:hashlib.sha256(t.encode('ascii')).hexdigest()
        files=[]
        for name,pred in [('control',[1,0]),('candidate',[1,2])]:
            artifact=root/(name+'.pb');artifact.write_bytes(('unit fixture '+name).encode('ascii'))
            report={'candidate_id':name,'evidence':'measured','evaluation_mode':'fixture_whole_hybrid','backend':'native_fixneuron',
                'dataset_sha256':fingerprint('dataset'),'preprocess_sha256':fingerprint('preprocess'),'fixed_artifacts_sha256':fingerprint('fixed'),
                'input_range_policy':'error','deployable':True,'sample_count':2,'correct_images':sum(a==b for a,b in zip([1,2],pred)),
                'accuracy':sum(a==b for a,b in zip([1,2],pred))/2.,'artifacts':{artifact.name:pp.sha(artifact)},
                'rows':[{'sample_id':str(i),'label':label,'prediction':p} for i,(label,p) in enumerate(zip([1,2],pred))]}
            path=root/(name+'.json');save(path,report);files.append(path.name)
        cfg={'evaluation_mode':'fixture_whole_hybrid','backend':'native_fixneuron','baseline_id':'control','reports':files}
        save(root/'ranking.json',cfg);return root/'ranking.json'

    def test_whole_chain_selection_portable_provenance_and_stale_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.reports(root);result=rc.rank(cfg,root/'out/selection.json')
            self.assertEqual(result['candidate_id'],'candidate')
            ag.check_selection(root/'out/selection.json',root/'candidate.pb')
            (root/'candidate.pb').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):ag.check_selection(root/'out/selection.json',root/'candidate.pb')

    def test_mixed_scope_or_backend_cannot_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.reports(root);candidate=pp.load_json(root/'candidate.json')
            candidate['evaluation_mode']='front_only';save(root/'candidate.json',candidate)
            with self.assertRaisesRegex(ValueError,'Wrong backend or evaluation scope'):rc.rank(cfg,root/'out.json')

def real_case(project,out):
    # Explicitly optional evidence run; no hardcoded dependency in either skill.
    sys.path.insert(0,str(project/'hybrid_split'));from common import ImageReader
    source=project/'modle/driver/pb_file/mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb'
    if not source.exists():source=project/'modle/pb_file/mobilevitv2_075_10cls_relu_deploy_b1_gnfold.pb'
    images=sorted((project/'modle/driver/Images').rglob('*.jpg'))
    if not source.exists() or not images:raise ValueError('Retained case source/images missing')
    samples=out/'samples';samples.mkdir(parents=True);reader=ImageReader()
    try:
        for i,index in enumerate(np.linspace(0,len(images)-1,min(10,len(images)),dtype=int)):
            np.savez(str(samples/('sample'+str(i)+'.npz')),**{'input_1:0':reader.read(images[int(index)])})
    finally:reader.close()
    cut='stack3_block2_pre_2_conv/Conv2D:0';output='predictions/BiasAdd:0'
    plan={'model_inputs':['input_1:0'],'model_outputs':[output],'parts':[
        {'name':'front','device':'dpu','require_static_nhwc4':True,'inputs':[{'tensor':'input_1:0','placeholder':'input_1'}],'outputs':[cut]},
        {'name':'tail','device':'ps','inputs':[{'tensor':cut,'placeholder':'ps_features'}],'outputs':[output]}]}
    save(out/'plan.json',plan);pp.split(source,out/'plan.json',out/'parts')
    ne.export(out/'parts/tail.pb','ps_features:0',output,out/'numpy/tail')
    result=pp.verify(source,out/'parts',samples,out/'case_verification.json',atol=1e-4,rtol=1e-4,numpy_dir=out/'numpy')
    elf=project/'hybrid_split/cle/artifacts/compile/dpu_mobilevitv2_075_front_cle.elf'
    so=project/'hybrid_mv2_dpu_20261031/deploy_in_board/model/libdpumodelmobilevitv2_075_front_cle.so'
    if elf.exists() and so.exists():
        matched=ag.sections(elf,1,'mobilevitv2_075_front_cle')==ag.sections(so,3,'mobilevitv2_075_front_cle')
        if not matched:raise ValueError('Existing case ELF/SO payload mismatch')
        result['retained_elf_so']={'matched':True,'scope':'Existing binary payload inspection only; no compiler/linker or DPU executed'}
    pairs={
        'front':(project/'hybrid_mv2_dpu_20261031/front_reference','input_1:0'),
        'mv2_4':(project/'hybrid_mv2_dpu_20261031/scheme_a/control/build/mv2_4/quantize','mv2_4_features:0'),
        'mv2_5':(project/'hybrid_mv2_dpu_20261031/scheme_a/control/build/mv2_5/quantize','mv2_5_features:0')}
    result['retained_quant_boundaries']={}
    for name,(directory,input_name) in pairs.items():
        if (directory/'quantize_eval_model.pb').exists() and (directory/'deploy_model.pb').exists():
            audit=qb.run(directory/'quantize_eval_model.pb',directory/'deploy_model.pb',input_name,
                         samples=samples if name=='front' else None)
            if not audit['consistent']:raise ValueError('Case input metadata inconsistent: '+name)
            result['retained_quant_boundaries'][name]=audit
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--real-case-root',type=Path);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();suite=unittest.TestSuite()
    for cls in (SplitTests,CleTests):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'passed':result.wasSuccessful(),'synthetic_tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'scope':'Host regression: numeric graph transformations and provenance, not native INT8 or board validation'}
    if a.real_case_root and result.wasSuccessful():
        with tempfile.TemporaryDirectory() as tmp:report['retained_case']=real_case(a.real_case_root,Path(tmp))
    pp.save_json(a.report,report)
    if not result.wasSuccessful():raise SystemExit(1)
