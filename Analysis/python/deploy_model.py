import os
import argparse

parser = argparse.ArgumentParser(description='Deploy keras model.')
parser.add_argument('--input', required=True, type=str, help="Input Keras model")
parser.add_argument('--output', required=False, type=str, default=None, help="Output Protocol Buffers file")
parser.add_argument('--check-nans', action='store_true')
args = parser.parse_args()

import tensorflow as tf
import numpy as np
from tensorflow.python.framework.graph_io import write_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.tools.graph_transforms import TransformGraph
from keras import backend as K
from common import LoadModel

config = tf.ConfigProto(intra_op_parallelism_threads=2,
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1, 'GPU' : 0})

session = tf.Session(config=config)
K.set_session(session)

K.set_learning_phase(0)
model = LoadModel(args.input, False)
print("Model loaded")
def node_names(nodes):
    return [ node.name.split(':')[0] for node in nodes ]
#print(model.inputs[0].name)
#print(model.outputs[0].name)
#raise RuntimeError("stop")

input_nodes = node_names(model.inputs)
output_nodes = node_names(model.outputs)

print("Inputs:", input_nodes)
print("Outputs:", output_nodes)
#input_nodes = ["main_input"]#  [model.inputs[0].name]
#output_nodes = ["main_output/Softmax"]#["output_node"]
#node_wrapper = tf.identity(model.outputs[0], name=output_nodes[0])

with K.get_session() as sess:

    if args.check_nans:
        for layer in model.layers:
            for weight in layer.weights:
                w = sess.run(tf.reshape(weight.value(), [-1]))
                if np.any(np.isnan(w)):
                    raise RuntimeError('Some weights in layer "{}/{}" are NaN.'.format(layer.name, weight.name))

    #final_graph = sess.graph.as_graph_def()
    const_graph = convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_nodes)
    #final_graph = const_graph
    transforms = [
        "strip_unused_nodes",
        "remove_nodes(op=Identity, op=CheckNumerics)",
        "fold_constants(ignore_errors=true)",
        "fold_batch_norms",
        #"quantize_weights"
    ]
    final_graph = TransformGraph(const_graph, input_nodes, output_nodes, transforms)

if args.output is None:
    input_base = os.path.basename(args.input)
    out_dir = '.'
    out_file = os.path.splitext(input_base)[0] + ".pb"
else:
    out_dir, out_file = os.path.split(args.output)
write_graph(final_graph, out_dir, out_file, as_text=False)
