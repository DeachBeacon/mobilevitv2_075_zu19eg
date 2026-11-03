"""Small TF1.15 helpers; no board, model or user-specific paths."""
from pathlib import Path
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import tensorflow as tf
tf1=tf.compat.v1
def import_graph(gd):
    graph=tf.Graph()
    with graph.as_default():tf1.import_graph_def(gd,name='')
    return graph
def cpu_config():return tf1.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)
def read_graph(path):
    gd=tf1.GraphDef();gd.ParseFromString(Path(path).read_bytes())
    if any(n.op=='FixNeuron' or 'wpos' in n.attr for n in gd.node):raise ValueError('Original float PB required, not eval/deploy PB or SO')
    graph=import_graph(gd)
    stateful=[op.name for op in graph.get_operations() if op.op_def.is_stateful]
    if stateful:raise ValueError('Frozen stateless inference graph required: '+str(stateful[:10]))
    return gd,graph
