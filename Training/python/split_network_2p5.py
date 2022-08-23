import os
import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
import cmsml

import sys ; sys.path.append('2018v1')
from Training_v0p1 import *
from DataLoader import DataLoader
from common import LoadModel

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import argparse
parser = argparse.ArgumentParser('''The script splits the DeepTauv2p5 into a inner, outer and core model.
Inner: from inner grid input to before merging with flat and outer
Outer: from outer grid input to before mergint with flat and inner
Core: from flat features+inner slice output + outer slice output to final prediction''')
parser.add_argument('--input'   , required=True, help='path to the input model')
parser.add_argument('--output'  , required=True, help='output directory to store model files and summaries')
parser.add_argument('--scaling' , required=True, help='path to the scaling yaml file')  # probably only needed to load the network setup
parser.add_argument('--training', required=True, help='path to the training yaml file')

args = parser.parse_args()

if not os.path.exists(args.output):
  os.makedirs(args.output)

with open(args.training, 'r') as training_file:
  training_cfg = yaml.unsafe_load(training_file)

dataloader = DataLoader(training_cfg, args.scaling)
net_config = dataloader.get_net_config()
aliases = {}

def test_prediction(model, reference):
  ''' compare the predictions of model (the slice) and reference (the full model).
  First create a sub-model from reference which stops at model inputs. Then, get predictions and use those as input to model.
  Compare model and reference predictions using consistent inputs and outputs.
  '''
  i_tensors = [tf.random.uniform(shape=[1]+l.shape.as_list()[1:], seed=2022) for l in reference.inputs]
  i_model = keras.Model(inputs=reference.inputs, outputs=[reference.get_layer(l.name.split('/')[0]).output for l in model.inputs])
  i_pred  = []
  for i, pred in enumerate(i_model.predict(i_tensors)):
    p_shape = pred.shape[1:]
    m_shape = tuple(model.inputs[i].shape.as_list()[1:])
    if p_shape==m_shape:
      i_pred.append(pred)
    elif len(p_shape)==len(m_shape)==3 and p_shape[2]==m_shape[2]:
      i_pred.append(pred.reshape((p_shape[0]*p_shape[1], 1, 1, p_shape[2])))
    else:
      raise RuntimeError("Invalid input shape for {}".format(model.name))
  o_model = keras.Model(inputs=reference.inputs, outputs=[reference.get_layer(l.name.split('/')[0]).output for l in model.outputs])
  ref_pred = o_model.predict(i_tensors)
  mod_pred =  model.predict(i_pred).reshape(ref_pred.shape)
  return np.max(np.absolute(np.subtract(ref_pred, mod_pred)))<1e-4

def copy_weights(model, target):
  ''' copy weights from model to target.
  Check for consistency and NaNs.
  '''
  for layer in target.layers:
    if 'Input' in layer.name or 'input' in layer.name: continue
    assert layer.name in [l.name for l in model.layers], "Layer '{}' of model '{}' not found in the full model configuration".format(layer.name, target.name)
    weights = model.get_layer(layer.name).get_weights()
    for x in weights:
      assert not np.any(np.isnan(x)), "Layer '{}' has NaN weights".format(layer.name)
      assert not np.any(np.isinf(x)), "Layer '{}' has Inf weights".format(layer.name)
    try:
      layer.set_weights(weights)
    except ValueError as _:
      try:
        r_weights = [w.reshape(layer.weights[i].shape) for i, w in enumerate(weights)]
        layer.set_weights(r_weights)
      except:
        raise RuntimeError("Weights from layer {} could not be loaded into the sliced model and could not be reshaped")

def save_to_graph(model, path):
  ''' use cmsml to save frozen graphs a la tf1
  '''
  cmsml.tensorflow.save_graph(path+'/'+model.name+'.pb', model, variables_to_constants=True)

def CoreModel(inner, outer):
  ''' The core model acting on flat features and the outputs of the convolutions, as defined in the training file
  '''
  global aliases
  tau_net_setup = NetSetup1D(**net_config.tau_net)
  dense_net_setup = NetSetup1D(**net_config.dense_net)
  conv_2d_net_setup = NetSetupConv2D(**net_config.conv_2d_net)
  input_layers = []
  high_level_features = []

  if net_config.n_tau_branches > 0:
    input_layer_tau = Input(name="input_tau", shape=(net_config.n_tau_branches,))
    input_layers.append(input_layer_tau)
    tau_net_setup.ComputeLayerSizes(net_config.n_tau_branches)
    processed_tau = reduce_n_features_1d(input_layer_tau, tau_net_setup, 'tau', net_config.first_layer_reg)
    high_level_features.append(processed_tau)

  for loc in ['inner', 'outer']:
    current_grid_size = net_config.n_cells[loc]
    n_inputs = inner.shape.as_list()[3] if loc == 'inner' else outer.shape.as_list()[3]
    n = 1
    lname = inner.name.split('/')[0] if loc=='inner' else outer.name.split('/')[0]
    lshape = inner.shape.as_list()[-1] if loc == 'inner' else outer.shape.as_list()[-1]
    prev_layer = Input(name=lname, shape=(net_config.n_cells[loc], net_config.n_cells[loc], lshape))
    aliases[lname] = 'input_'+loc
    input_layers.append(prev_layer)
    while current_grid_size > 1:
      win_size = min(current_grid_size, conv_2d_net_setup.window_size)
      n_filters = get_n_filters_conv2d(n_inputs, current_grid_size, win_size, conv_2d_net_setup.reduction_rate)
      prev_layer = conv_block(prev_layer, n_filters, (win_size, win_size), conv_2d_net_setup,
                              "{}_all_{}x{}".format(loc, win_size, win_size), n)
      n += 1
      current_grid_size -= win_size - 1
      n_inputs = n_filters
    
    cells_flatten = Flatten(name="{}_cells_flatten".format(loc))(prev_layer)
    high_level_features.append(cells_flatten)
  
  features_concat = Concatenate(name="features_concat", axis=1)(high_level_features)
  dense_net_setup.ComputeLayerSizes(features_concat.shape.as_list()[1])
  final_dense = reduce_n_features_1d(features_concat, dense_net_setup, 'final')
  output_layer = Dense(net_config.n_outputs, name="final_dense_last", kernel_initializer=dense_net_setup.kernel_init)(final_dense)
  softmax_output = Activation("softmax", name="main_output")(output_layer)
  return keras.Model(input_layers, softmax_output, name='core')

def ConvDenseAlias(loc):
  ''' The 1x1 conv. model acting on the PFCand grid, as defined in the training file.
  Replace the convolution with dense layers acting on the single cells (through the input features)
  '''
  comp_net_setup = NetSetup2D(**net_config.comp_net)
  comp_merge_net_setup = NetSetup2D(**net_config.comp_merge_net)

  conv_2d_net_setup = NetSetupConv2D(**net_config.conv_2d_net)
  reduced_inputs = []
  input_layers   = []
  for comp_id in range(len(net_config.comp_names)):
    comp_name = net_config.comp_names[comp_id]
    n_comp_features = net_config.n_comp_branches[comp_id]
    input_layer_comp = Input(name="input_{}_{}".format(loc, comp_name), shape=(1,1,n_comp_features))
    input_layers.append(input_layer_comp)
    comp_net_setup.ComputeLayerSizes(n_comp_features)
    # here we replace 2d with 1d (1x1 -> dense)
    reduced_comp = reduce_n_features_1d(input_layer_comp, comp_net_setup, "{}_{}".format(loc, comp_name), net_config.first_layer_reg, basename='conv')
    reduced_inputs.append(reduced_comp)

  if len(net_config.comp_names) > 1:
    conv_all_start = Concatenate(name="{}_cell_concat".format(loc), axis=3)(reduced_inputs)
    comp_merge_net_setup.ComputeLayerSizes(conv_all_start.shape.as_list()[3])
    # here we replace 2d with 1d (1x1 -> dense)
    prev_layer = reduce_n_features_1d(conv_all_start, comp_merge_net_setup, "{}_all".format(loc), basename='conv')
  else:
    prev_layer = reduced_inputs[0]

  return keras.Model(input_layers, prev_layer, name=loc)

def parse_aliases(model):
  for k, v in aliases.items():
    if not k in [l.name for l in model.layers]:
      continue
    model.get_layer(k)._name = v
    model.get_layer(v).output._name = model.get_layer(v).output.name.replace(k, v)

full_model = LoadModel(args.input)

inner_model = ConvDenseAlias('inner')
outer_model = ConvDenseAlias('outer')
core_model  = CoreModel(inner=inner_model.output, outer=outer_model.output)

copy_weights(model=full_model, target=inner_model)
copy_weights(model=full_model, target=outer_model)
copy_weights(model=full_model, target=core_model )

assert test_prediction(model=inner_model, reference=full_model), "Error: '{}' model predictions differs from the full model ones".format(inner_model.name)
assert test_prediction(model=outer_model, reference=full_model), "Error: '{}' model predictions differs from the full model ones".format(outer_model.name)
assert test_prediction(model=core_model , reference=full_model), "Error: '{}' model predictions differs from the full model ones".format(core_model.name)

parse_aliases(inner_model)
parse_aliases(outer_model)
parse_aliases(core_model)

# workaround: conversion to tf graph implies that graph operations get the prefix <model_name> in their names
inner_model._name = ''
outer_model._name = ''
core_model ._name = ''
full_model ._name = ''

with open(args.output+"/inner_summary.txt", "w") as smr:
  inner_model.summary(print_fn=lambda x: smr.write(x+'\n'))
with open(args.output+"/outer_summary.txt", "w") as smr:
  outer_model.summary(print_fn=lambda x: smr.write(x+'\n'))
with open(args.output+"/core_summary.txt", "w") as smr:
  core_model.summary(print_fn=lambda x: smr.write(x+'\n'))
with open(args.output+"/full_summary.txt", "w") as smr:
  full_model.summary(print_fn=lambda x: smr.write(x+'\n'))

save_to_graph(inner_model, args.output)
save_to_graph(outer_model, args.output)
save_to_graph(core_model , args.output)
save_to_graph(full_model , args.output)

print("Consistency checks are OK. Frozen graphs saved in {}".format(args.output))