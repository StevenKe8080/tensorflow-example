from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # 输入层
  # 重新生成X个4纬张量：[batch_size, width, height, channels] 数量，宽度，高度，图片通道
  # MNIST是28x28像素只有单通道的图片
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # 卷积层 #1
  # 用5x5的过滤器和ReLU激活函数计算32个特征
  # 添加填充以保留宽度和高度
  # 输入 Tensor 形状: [batch_size, 28, 28, 1]
  # 输出 Tensor 形状: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # 池化层 #1
  # 第一个最大池化层使用2x2过滤和步幅2
  # 输入 Tensor 形状: [batch_size, 28, 28, 32]
  # 输出 Tensor 形状: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # 卷积层 #2
  # 用5x5的过滤器和ReLU激活函数计算64个特征
  # 添加填充以保留宽度和高度
  # 输入 Tensor 形状: [batch_size, 14, 14, 32]
  # 输出 Tensor 形状: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # 池化层  #2
  # 第二个最大池化层使用2x2过滤和步幅2
  # 输入 Tensor 形状: [batch_size, 14, 14, 64]
  # 输出 Tensor 形状: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # tensor扁平化
  # 输入 Tensor 形状: [batch_size, 7, 7, 64]
  # 输出 Tensor 形状: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # 全连接层
  # 全连接层有1024个神经元
  # 输入 Tensor 形状: [batch_size, 7 * 7 * 64]
  # 输出 Tensor 形状: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # 添加dropout操作：百分之60的元素被保留
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits层
  # 输入 Tensor 形状: [batch_size, 1024]
  # 输出 Tensor 形状: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # 生成预测，用于预测和评估模式
      "classes": tf.argmax(input=logits, axis=1),
      # 添加softmax_tensor，用于预测
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # 计算损失用于训练和评估模式
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # 批准训练操作器，用于训练模式
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # 添加评估指标,用于评估模式
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # 读取训练和评估数据
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # 返回 np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # 返回 np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # 创建Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="model")

  #为预测设置日志记录
  # 用“probabilities”标签记录“Softmax”张量中的值
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # 训练模型
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # 评估模型打印结果
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()