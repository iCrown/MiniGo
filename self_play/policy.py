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
    def __init__(self, logdir=None, read_file=None):
        self.lr = 1e-3
        self.features = features.DEFAULT_FEATURES
        self.n_input_planes = sum(f.planes for f in self.features)
        self.n_planes = 64
        self.n_layers = 5
        self.batch_size = 128
        self.log_per_epoch = 40

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
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

    def build_policy_network_op(self, scope="PolicNetwork"):
        with tf.variable_scope(scope, reuse=False):
            hidden = tf.contrib.layers.conv2d(self.observation_placeholder, num_outputs=self.n_planes, kernel_size=5, stride=1)
            for i in range(self.n_layers):
                with tf.variable_scope("layer"+str(i), reuse=False):
                    hidden = tf.contrib.layers.conv2d(hidden, num_outputs=self.n_planes, kernel_size=3, stride=1)
            hidden_final = tf.contrib.layers.conv2d(hidden, num_outputs=1, kernel_size=1, stride=1, activation_fn=None)
            self.logits = tf.contrib.layers.flatten(hidden_final)

    def add_loss_acc_op(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.action_placeholder))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.action_placeholder), tf.float32))

    def add_optimizer_op(self):
        learning_rate = tf.train.exponential_decay(self.lr, self.global_step, 10000, 0.92, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=self.global_step)

    def initialize_logging(self, logdir):
        if logdir is not None:
            self.test_summary_writer = tf.summary.FileWriter(os.path.join(logdir, "test"), self.session.graph)
            self.training_summary_writer = tf.summary.FileWriter(os.path.join(logdir, "train"), self.session.graph)
            self.test_stats = StatisticsCollector()
            self.training_stats = StatisticsCollector()

    def initialize_variables(self, read_file=None):
        self.session.run(tf.global_variables_initializer())
        if read_file is not None:
            self.saver.restore(self.session, read_file)

    def get_global_step(self):
        return self.session.run(self.global_step)

    def save_variables(self, save_file):
        if save_file is not None:
            print("Saving checkpoint to %s" % save_file, file=sys.stderr)
            self.saver.save(self.session, save_file)

    def train(self, train_data, test_data):
        num_minibatches = train_data.data_size // self.batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = train_data.get_batch(self.batch_size)
            _, accuracy, cost = self.session.run(
                [self.train_op, self.accuracy, self.loss],
                feed_dict={self.observation_placeholder: batch_x, self.action_placeholder: np.argmax(batch_y, 1)})
            self.training_stats.report(accuracy, cost)

            if i % (num_minibatches // self.log_per_epoch) == 0:
                # log train
                avg_accuracy, avg_cost, acc_cost_summaries = self.training_stats.collect()
                global_step = self.get_global_step()
                print("===Step: %d" % global_step)
                print("===train===: accuracy: %g; cost: %g" % (avg_accuracy, avg_cost))
                if self.training_summary_writer is not None:
                    self.training_summary_writer.add_summary(acc_cost_summaries, global_step)
                # log eval
                self.test(test_data)

    def run(self, position):
        'Return a sorted list of (probability, move) tuples'
        processed_position = features.extract_features(position, features=self.features)
        probabilities = self.session.run(self.logits, feed_dict={self.observation_placeholder: processed_position[None, :]})[0]
        return probabilities.reshape([go.N, go.N])

    def test(self, test_data):
        num_minibatches = test_data.data_size // self.batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = test_data.get_batch(self.batch_size)
            accuracy, cost = self.session.run(
                [self.accuracy, self.loss],
                feed_dict={self.observation_placeholder: batch_x, self.action_placeholder: np.argmax(batch_y, 1)})
            self.test_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, acc_cost_summaries = self.test_stats.collect()
        global_step = self.get_global_step()
        print("===test===: accuracy: %g; cost: %g" % (avg_accuracy, avg_cost))
        if self.test_summary_writer is not None:
            self.test_summary_writer.add_summary(acc_cost_summaries, global_step)

class StatisticsCollector(object):
    graph = tf.Graph()
    with tf.device("/cpu:0"), graph.as_default():
        accuracy = tf.placeholder(tf.float32, [])
        cost = tf.placeholder(tf.float32, [])
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        cost_summary = tf.summary.scalar("cost", cost)
        acc_cost_summaries = tf.summary.merge([accuracy_summary, cost_summary], name="acc_cost_summaries")
    session = tf.Session(graph=graph)

    def __init__(self):
        self.accuracies = []
        self.costs = []

    def report(self, accuracy, cost):
        self.accuracies.append(accuracy)
        self.costs.append(cost)

    def collect(self):
        avg_acc = sum(self.accuracies) / len(self.accuracies)
        avg_cost = sum(self.costs) / len(self.costs)
        self.accuracies = []
        self.costs = []
        summary = self.session.run(self.acc_cost_summaries,
            feed_dict={self.accuracy:avg_acc, self.cost: avg_cost})
        return avg_acc, avg_cost, summary
