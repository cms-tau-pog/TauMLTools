import os
import argparse

parser = argparse.ArgumentParser(description='Deploy keras model.')
parser.add_argument('--input', required=True, type=str, help="Output Protocol Buffers file")
parser.add_argument('--output', required=False, type=str, default=None, help="Output Protocol Buffers file")
args = parser.parse_args()

import tensorflow as tf
from tensorflow.python.framework.graph_io import write_graph
from tensorflow.tools.graph_transforms import TransformGraph

input_nodes = ["main_input"]#  [model.inputs[0].name]
output_nodes = ["main_output/Softmax"]#["output_node"]
#node_wrapper = tf.identity(model.outputs[0], name=output_nodes[0])

def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

graph = load_graph(args.input)
sess = tf.Session(graph=graph)
ops = sess.graph.get_operations()

transforms = [
    "quantize_weights"
]
final_graph = TransformGraph(sess.graph.as_graph_def(), input_nodes, output_nodes, transforms)

if args.output is None:
    input_base = os.path.basename(args.input)
    out_dir = '.'
    out_file = os.path.splitext(input_base)[0] + "_quantized.pb"
else:
    out_dir, out_file = os.path.split(args.output)
write_graph(final_graph, out_dir, out_file, as_text=False)
