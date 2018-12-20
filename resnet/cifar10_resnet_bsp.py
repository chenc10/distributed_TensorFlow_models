from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
import resnet_model

import cifar10
from tensorflow.python.client import timeline

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

INITIAL_LEARNING_RATE = 0.08      # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EPOCHS_PER_DECAY = 32.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5
_WEIGHT_DECAY = 2e-3

def train():
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    print ('PS hosts are: %s' % ps_hosts)
    print ('Worker hosts are: %s' % worker_hosts)
    configP=tf.ConfigProto()
    server = tf.train.Server({'ps': ps_hosts, 'worker': worker_hosts},
                             job_name = FLAGS.job_name,
                             task_index=FLAGS.task_id,
			     config=configP)

    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_id == 0)
    if is_chief:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    device_setter = tf.train.replica_device_setter(ps_tasks=len(ps_hosts))
    with tf.device('/job:worker/task:%d' % FLAGS.task_id):
        with tf.device(device_setter):

            """Prepare Input"""
            global_step = tf.Variable(0, trainable=False)
	    decay_steps = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * NUM_EPOCHS_PER_DECAY /FLAGS.batch_size
	    batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
            with tf.device('/cpu:0'):
                images, labels = cifar10.distorted_inputs(batch_size)
            inputs = tf.reshape(images, [-1, _HEIGHT, _WIDTH, _DEPTH])

            """Inference"""
	    with tf.variable_scope('root', partitioner=tf.fixed_size_partitioner(len(ps_hosts), axis=0)):
            	network = resnet_model.cifar10_resnet_v2_generator(FLAGS.resnet_size, _NUM_CLASSES)
            logits = network(inputs, True)
            labels = tf.cast(labels, tf.int64)
            correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy_op = tf.reduce_mean(correct_prediction)

            """Loss"""
            labels = tf.one_hot(labels, 10, 1, 0)
            cross_entropy = tf.losses.softmax_cross_entropy(
                logits=logits, 
                onehot_labels=labels)
            loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

            """Define Optimization"""
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE * len(worker_hosts),
                                            global_step,
                                            decay_steps,
                                            LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            opt = tf.train.GradientDescentOptimizer(lr)
            # Track the moving averages of all trainable variables.
            exp_moving_averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=len(worker_hosts),
                total_num_replicas=len(worker_hosts),
                variable_averages=exp_moving_averager,
                variables_to_average=variables_to_average)
            # Compute gradients with respect to the loss.
            grads = opt.compute_gradients(loss) 
            apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
            with tf.control_dependencies([apply_gradients_op]):
                train_op = tf.identity(loss, name='train_op')

            """Sychronization Management"""
            if is_chief:
                chief_queue_runners = [opt.get_chief_queue_runner()]
                init_tokens_op = opt.get_init_tokens_op()
            saver = tf.train.Saver(max_to_keep=1)
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir=FLAGS.train_dir,
				     init_op=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()),
                                     summary_op=None,
                                     global_step=global_step,
                                     saver=saver,
				     recovery_wait_secs=1,
                                     save_model_secs=60)
            tf.logging.info('%s Supervisor' % datetime.now())

            """Train CIFAR-10 for a number of steps."""
   	    sess_config = tf.ConfigProto(allow_soft_placement=True,
   	                                 log_device_placement=FLAGS.log_device_placement)
   	    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)

            if is_chief:
               sv.start_queue_runners(sess, chief_queue_runners)
               sess.run(init_tokens_op)

	    batch_size_num = FLAGS.batch_size
            for step in range(init_global_step, FLAGS.max_steps):
                step_start_time = time.time()
      		run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      		run_metadata = tf.RunMetadata()
                num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size_num
                decay_steps_num = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                _, loss_value, gs = sess.run([train_op, loss, global_step], feed_dict={batch_size: batch_size_num},  options=run_options, run_metadata=run_metadata)

                duration = time.time() - step_start_time
                num_examples_per_step = batch_size_num
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ("time: " + str(time.time()) + '; %s: step %d (gs %d), loss= %.2f (%.1f samples/s; %.3f s/batch)')
                tf.logging.info(format_str % (datetime.now(), step, gs, loss_value, examples_per_sec, sec_per_batch))

                """Do evaluation on accuracy (this is not testset evaluation)"""
                if step % 200 == 0:
                    accuracy = sess.run(accuracy_op, feed_dict={batch_size: 10000})
                    tf.logging.info('evaluation: step - '+str(step)+'; accuracy: '+ str(accuracy))
                    


def main(argv=None):
    cifar10.maybe_download_and_extract()
    train()

if __name__ == '__main__':
    tf.app.run()
