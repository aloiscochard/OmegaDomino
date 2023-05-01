import os
import sys

import tensorflow as tf

if len(sys.argv) >= 2:
    if sys.argv[1] == "EVAL":
        mode = tf.estimator.ModeKeys.EVAL
    elif sys.argv[1] == "PREDICT":
        mode = tf.estimator.ModeKeys.PREDICT
    elif sys.argv[1] == "TRAIN":
        mode = tf.estimator.ModeKeys.TRAIN
    else:
        sys.exit(1)
    print(mode)


def graph_dump(name="", suffix=""):
    if name == "":
        name = os.path.basename(__file__)[:-3]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    graph_def = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).graph.as_graph_def()

    tf.train.write_graph(graph_def, './', '%s%s.pbtxt'%(name, suffix), as_text = True)
    tf.train.write_graph(graph_def, './', '%s%s.pb'%(name, suffix), as_text = False)

def network_dump(name=""):
    if mode == tf.estimator.ModeKeys.EVAL:
        suffix = "-eval"
    elif mode == tf.estimator.ModeKeys.PREDICT:
        suffix = "-predict"
    elif mode == tf.estimator.ModeKeys.TRAIN:
        suffix = "-train"
    else:
        sys.exit(1)

    graph_dump(name, suffix)
