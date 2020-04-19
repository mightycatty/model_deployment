import tensorflow as tf
import logging

from tf_graph_toolkit import get_graphdef_wrapper, _auto_inputs_outputs_detect, graph_optimization

logger = logging.getLogger('IR Converter')
logger.setLevel(logging.INFO)

@get_graphdef_wrapper
def convert_frozen_pb_to_onnx(frozen_pb_or_graph_def, opset=9, tf_graph_optimization=True, input_shape=None, name=None):
    try:
        from tf2onnx.tfonnx import process_tf_graph, tf_optimize
        from tf2onnx import constants, loader, logging, utils, optimizer
    except Exception as e:
        logger.error('import tf2onnx error, "pip install tf2onnx"')
        exit(0)

    graph_def = frozen_pb_or_graph_def
    if isinstance(frozen_pb_or_graph_def, str):
        model_path = frozen_pb_or_graph_def
        output_dir = model_path.replace('.pb', '.onnx')
    else:
        model_path = 'graphdef_buffer'
        assert name, 'name should be give to export an .onnx when converting from a graphdef buffer'
        output_dir = '{}.onnx'.format(name)
    inputs, outputs = _auto_inputs_outputs_detect(graph_def)
    shape_override = {}
    if input_shape:
        assert isinstance(input_shape, list), 'input_shape item need to be list, each for dims of a input tensor'
        for idx, item in enumerate(input_shape):
            shape_override[inputs[idx]] = item
            # graph optimizatin with tf_graph_transform
    if tf_graph_optimization:
        graph_def = graph_optimization(graph_def)
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.Session(graph=tf_graph):
        g = process_tf_graph(tf_graph,
                             continue_on_error=False,
                             target='',
                             opset=opset,
                             custom_op_handlers={},
                             extra_opset=[],
                             shape_override=shape_override,
                             input_names=inputs,
                             output_names=outputs,
                             inputs_as_nchw=None)
    # graph optimization with onnx optimizer
    onnx_graph = optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model("converted from {}".format(model_path))

    # write onnx graph
    logger.info("")
    logger.info("Successfully converted TensorFlow model %s to ONNX", model_path)
    outputs = model_path.replace('.pb', '.onnx')
    utils.save_protobuf(output_dir, model_proto)
    logger.info("ONNX model is saved at %s", output_dir)