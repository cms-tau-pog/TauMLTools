import os
import argparse

parser = argparse.ArgumentParser(description='Deploy keras model.')
parser.add_argument('--input', required=True, type=str, help="Output Protocol Buffers file")
parser.add_argument('--output', required=True, type=str, help="Output Protocol Buffers file")
args = parser.parse_args()

import tensorflow as tf

def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

graph = load_graph(args.input)
sess = tf.Session(graph=graph)

writer = tf.summary.FileWriter(args.output, sess.graph)
