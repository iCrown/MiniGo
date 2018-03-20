'''
Neural network architecture.
The input to the policy network is a 19 x 19 x 48 image stack consisting of
48 feature planes. The first hidden layer zero pads the input into a 23 x 23
image, then convolves k filters of kernel size 5 x 5 with stride 1 with the
input image and applies a rectifier nonlinearity. Each of the subsequent
hidden layers 2 to 12 zero pads the respective previous hidden layer into a
21 x 21 image, then convolves k filters of kernel size 3 x 3 with stride 1,
again followed by a rectifier nonlinearity. The final layer convolves 1 filter
of kernel size 1 x 1 with stride 1, with a different bias for each position,
and applies a softmax function. The match version of AlphaGo used k = 192
filters; Fig. 2b and Extended Data Table 3 additionally show the results
of training with k = 128, 256 and 384 filters.

The input to the value network is also a 19 x 19 x 48 image stack, with an
additional binary feature plane describing the current colour to play.
Hidden layers 2 to 11 are identical to the policy network, hidden layer 12
is an additional convolution layer, hidden layer 13 convolves 1 filter of
kernel size 1 x 1 with stride 1, and hidden layer 14 is a fully connected
linear layer with 256 rectifier units. The output layer is a fully connected
linear layer with a single tanh unit.
'''
import math
import os
import sys
import tensorflow as tf

import features
import go
import utils
import numpy as np

EPSILON = 1e-35

class PolicyNetwork(object):
    def __init__(self, scope, logdir=None, read_file=None):
        self.lr = 1e-3
        self.features = features.DEFAULT_FEATURES
        self.n_input_planes = sum(f.planes for f in self.features)
        self.n_planes = 64
        self.n_layers = 5
        self.batch_size = 128
        self.log_freq_minibatch = 10
        self.scope = scope
        self.lr_decay_step = 2000

        self.session = tf.Session()
        self.build()
        self.saver = tf.train.Saver()
        self.initialize_variables(read_file)
        self.initialize_logging(logdir)

    def build(self):
        self.add_placeholders_op()
        self.build_policy_network_op()
        self.add_loss_acc_op()
        self.add_optimizer_op()
        

    def add_placeholders_op(self):
        self.observation_placeholder = tf.placeholder(tf.float32, shape=[None, go.N, go.N, self.n_input_planes])
        self.action_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.rewards_placeholder = tf.placeholder(tf.float32, shape=[None])
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

    def build_policy_network_op(self):
        with tf.variable_scope(self.scope, reuse=False):
            hidden = tf.contrib.layers.conv2d(self.observation_placeholder, num_outputs=self.n_planes, kernel_size=5, stride=1)
            for i in range(self.n_layers):
                with tf.variable_scope("layer"+str(i), reuse=False):
                    hidden = tf.contrib.layers.conv2d(hidden, num_outputs=self.n_planes, kernel_size=3, stride=1)
            hidden_final = tf.contrib.layers.conv2d(hidden, num_outputs=1, kernel_size=1, stride=1, activation_fn=None)
            self.logits = tf.contrib.layers.flatten(hidden_final)
        self.logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.action_placeholder)
        


    def add_loss_acc_op(self):
        self.loss = tf.reduce_mean(self.logprob * self.rewards_placeholder)

    def add_optimizer_op(self):
        learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.lr_decay_step, 0.92, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)

    def initialize_logging(self, logdir):
        if logdir is not None:
            self.training_summary_writer = tf.summary.FileWriter(os.path.join(logdir, "train"), self.session.graph)
            self.training_stats = StatisticsCollector()

    def initialize_variables(self, read_file=None):
        self.session.run(tf.global_variables_initializer())
        if read_file is not None:
            self.saver.restore(self.session, read_file)

    def get_global_step(self):
        return self.session.run(self.global_step)

    def save_variables(self, save_file):
        if save_file is not None:
            print("Saved checkpoint to %s" % save_file, file=sys.stderr)
            self.saver.save(self.session, save_file)

    def train(self, observations, actions, rewards):
        _, cost = self.session.run(
            [self.train_op, self.loss],
            feed_dict={self.observation_placeholder: observations,
                       self.action_placeholder: actions,
                       self.rewards_placeholder: rewards})
        self.training_stats.report(cost)

        avg_cost, summary = self.training_stats.collect()
        global_step = self.get_global_step()
        print("global step: %d; cost: %g" % (global_step, avg_cost))
        if self.training_summary_writer is not None:
            self.training_summary_writer.add_summary(summary, global_step)


    def run(self, position):
        'Return a sorted list of (probability, move) tuples'
        processed_position = features.extract_features(position, features=self.features)
        probabilities = self.session.run(self.logits, feed_dict={self.observation_placeholder: processed_position[None, :]})[0]
        return probabilities.reshape([go.N, go.N])



class StatisticsCollector(object):
  graph = tf.Graph()
  with tf.device("/cpu:0"), graph.as_default():
    cost = tf.placeholder(tf.float32, [])
    cost_summary = tf.summary.scalar("cost", cost)
  session = tf.Session(graph=graph)

  def __init__(self):
    self.costs = []

  def report(self, cost):
    self.costs.append(cost)

  def collect(self):
    avg_cost = sum(self.costs) / len(self.costs)
    self.costs = []
    summary = self.session.run(self.cost_summary,
      feed_dict={self.cost: avg_cost})
    return avg_cost, summary
