# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to train Inception using multiple replicas with synchronous update.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import numpy as np
import tensorflow as tf
#from inception.imagenet_data import ImagenetData
#from inception import image_processing
#from inception import inception_model as inception
#from inception.slim import slim
from imagenet_data import ImagenetData
import image_processing
import inception_model as inception
from slim import slim
from tensorflow.python.client import timeline

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('protocol', 'grpc',
                           """Communication protocol to use in distributed """
                           """execution (default grpc) """)
tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/imagenet/train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')

RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def main(argv=None):
  ps_hosts = FLAGS.ps_hosts.split(',')
  worker_hosts = FLAGS.worker_hosts.split(',')
  tf.logging.info('PS hosts are: %s' % ps_hosts)
  tf.logging.info('Worker hosts are: %s' % worker_hosts)
  cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,'worker': worker_hosts})
  server = tf.train.Server({'ps': ps_hosts,'worker': worker_hosts},
      job_name=FLAGS.job_name,task_index=FLAGS.task_id,protocol=FLAGS.protocol)

  if FLAGS.job_name == 'ps':
    server.join()

  dataset = ImagenetData(subset=FLAGS.subset)
  assert dataset.data_files()
  is_chief = (FLAGS.task_id == 0)
  if is_chief:
    if not tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.MakeDirs(FLAGS.train_dir)

  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])

  with tf.device('/job:worker/task:%d' % FLAGS.task_id):
    with slim.scopes.arg_scope(
        [slim.variables.variable, slim.variables.global_step],
        device=slim.variables.VariableDeviceChooser(num_parameter_servers)):

      '''Prepare Input'''
      global_step = slim.variables.global_step()
      batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
      images, labels = image_processing.distorted_inputs(dataset,batch_size,
          num_preprocess_threads=FLAGS.num_preprocess_threads)
      num_classes = dataset.num_classes() + 1

      '''Inference'''
      logits = inception.inference(images, num_classes, for_training=True)

      '''Loss'''
      inception.loss(logits, labels, batch_size)
      losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
      losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      total_loss = tf.add_n(losses, name='total_loss')
      if is_chief:
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        with tf.control_dependencies([loss_averages_op]):
          total_loss = tf.identity(total_loss)

      '''Optimizer'''
      exp_moving_averager = tf.train.ExponentialMovingAverage(
          inception.MOVING_AVERAGE_DECAY, global_step)
      variables_to_average = (
          tf.trainable_variables() + tf.moving_average_variables())
      num_batches_per_epoch = (dataset.num_examples_per_epoch() / FLAGS.batch_size)
      decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay / num_workers)
      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate * num_workers,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True)
      opt = tf.train.RMSPropOptimizer(lr,RMSPROP_DECAY,
                                      momentum=RMSPROP_MOMENTUM,
                                      epsilon=RMSPROP_EPSILON)
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=num_workers,
          total_num_replicas=num_workers,
          variable_averages=exp_moving_averager,
          variables_to_average=variables_to_average)

      '''Train Operation'''
      batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
      assert batchnorm_updates, 'Batchnorm updates are missing'
      batchnorm_updates_op = tf.group(*batchnorm_updates)
      with tf.control_dependencies([batchnorm_updates_op]):
        total_loss = tf.identity(total_loss)
      naive_grads = opt.compute_gradients(total_loss) 
      grads = [(tf.scalar_mul(tf.cast(batch_size/FLAGS.batch_size, tf.float32), grad), var) for grad, var in naive_grads]
      apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
      with tf.control_dependencies([apply_gradients_op]):
        train_op = tf.identity(total_loss, name='train_op')

      '''Supervisor and Session''' 
      init_tokens_op = opt.get_init_tokens_op()
      saver = tf.train.Saver()
      init_op = tf.global_variables_initializer()
      sv = tf.train.Supervisor(is_chief=is_chief,
                               logdir=FLAGS.train_dir,
                               init_op=init_op,
                               summary_op=None,
                               global_step=global_step,
                               recovery_wait_secs=1,
                               saver=saver,
                               save_model_secs=FLAGS.save_interval_secs)
      tf.logging.info('%s Supervisor' % datetime.now())
      sess_config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement)
      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
      queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)

      '''Start Training'''
      sv.start_queue_runners(sess, queue_runners)
      tf.logging.info('Started %d queues for processing input data.',
                      len(queue_runners))
      chief_queue_runners = [opt.get_chief_queue_runner()]
      if is_chief:
        sv.start_queue_runners(sess, chief_queue_runners)
        sess.run(init_tokens_op)
        
      batch_size_num = FLAGS.batch_size
      for step in range(FLAGS.max_steps):
          start_time = time.time()
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          loss_value, gs = sess.run([train_op, global_step], feed_dict={batch_size: batch_size_num}, options=run_options, run_metadata=run_metadata)

          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

          duration = time.time() - start_time
          examples_per_sec = batch_size_num / float(duration)
          sec_per_batch = float(duration)
          format_str = ("time: " + str(time.time()) + '; %s: step %d (gs %d), loss= %.2f (%.1f samples/s; %.3f s/batch)')
          tf.logging.info(format_str % (datetime.now(), step, gs, loss_value, examples_per_sec, sec_per_batch))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
