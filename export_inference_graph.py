# coding=UTF-8
"""
四合一
将 tensorflow 训练过程中生成的文件(checkpoint,model-7800.data-00000-of-00001,model-7800.index, model-7800.meta)转成pb文件
"""


import tensorflow as tf
import os.path

MODEL_DIR = "/home/zhwpeng/project/p0305/forecast_extraction/abcft_algorithm_forecast_extraction/text_classify/c_model/checkpoints"
MODEL_NAME = "title_classify1.pb"
output_node_names = ['output/predictions']

checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)  # 检查目录下ckpt文件状态是否可用
input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径
output_graph = os.path.join(MODEL_DIR, MODEL_NAME)  # PB模型保存路径

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
    # saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # saver.restore(sess, tf.train.latest_checkpoint('.'))
    var_SRGAN_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)

    # Save the frozen graph
    with open(output_graph, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
        # print("%d ops in the final graph." % len(frozen_graph_def.node))



