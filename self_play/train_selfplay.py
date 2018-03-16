# -*- coding: UTF-8 -*-

import os
import sys
import time
import numpy as np
import tensorflow as tf
import os
import time

import features
import go
from tqdm import tqdm


class PG(object):
  def __init__(self):            
    # store hyper-params

    self.use_baseline = True
    self.normalize_advantage =False
    self.batch_size = 4096
    self.num_batches = 10000
    self.checkpoint_freq = 1000
    self.print_freq = 50
    self.lr_decay_step = 200
    self.gamma = 0.1
    self.lr = 1e-4
    self.lr_baseline = self.lr * 10
    self.n_input_planes = sum(f.planes for f in features.DEFAULT_FEATURES)

    self.build()
  
  def add_placeholders_op(self):
    self.observation_placeholder = tf.placeholder(tf.float32, shape=[None, go.N, go.N, self.n_input_planes])
    self.action_placeholder = tf.placeholder(tf.int32, shape=[None])
    self.advantage_placeholder = tf.placeholder(tf.float32, shape=[None])
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
  
  def build_policy_network_op(self, scope = "PolicNetwork"):
    with tf.variable_scope(scope, reuse=False):
      hidden = tf.contrib.layers.conv2d(self.observation_placeholder, num_outputs=64, kernel_size=5, stride=1)
      for i in range(5):
        with tf.variable_scope("layer"+str(i), reuse=False):
          hidden = tf.contrib.layers.conv2d(hidden, num_outputs=64, kernel_size=3, stride=1)
      hidden_final = tf.contrib.layers.conv2d(hidden, num_outputs=1, kernel_size=1, stride=1, activation_fn=None)
      self.action_logits = tf.contrib.layers.flatten(hidden_final)
    self.logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.action_logits, labels=self.action_placeholder)

  def add_loss_op(self):
    self.loss = tf.reduce_mean(self.logprob * self.advantage_placeholder)
  
  def add_optimizer_op(self):
    lr_decay = tf.train.exponential_decay(self.lr, self.global_step, self.lr_decay_step, 0.92, staircase=True)
    self.train_op = tf.train.AdamOptimizer(learning_rate=lr_decay).minimize(self.loss)
  
  def add_baseline_op(self, scope = "ValueNetwork"):
    with tf.variable_scope(scope, reuse=False):
      hidden = tf.contrib.layers.conv2d(self.observation_placeholder, num_outputs=64, kernel_size=5, stride=1)
      for i in range(5):
        with tf.variable_scope("layer"+str(i), reuse=False):
          hidden = tf.contrib.layers.conv2d(hidden, num_outputs=64, kernel_size=3, stride=1)
      hidden_final = tf.contrib.layers.conv2d(hidden, num_outputs=1, kernel_size=1, stride=1, activation_fn=None)
      flatten = tf.contrib.layers.flatten(hidden_final)
      output = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=1)
    self.baseline = tf.squeeze(output, axis=1)
    self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=[None])
    self.baseline_loss = tf.losses.mean_squared_error(self.baseline_target_placeholder, self.baseline, scope=scope)
    lr_baseline_decay = tf.train.exponential_decay(self.lr_baseline, self.global_step, self.lr_decay_step, 0.92, staircase=True)
    self.update_baseline_op = tf.train.AdamOptimizer(learning_rate=lr_baseline_decay).minimize(self.baseline_loss)
  
  def build(self):
    self.add_placeholders_op()
    self.build_policy_network_op()
    self.add_loss_op()
    self.add_optimizer_op()
    if self.use_baseline:
      self.add_baseline_op()

  def step(self, pos_cur):
    state = features.extract_features(pos_cur)
    move_probs = self.sess.run(self.action_logits, feed_dict={self.observation_placeholder: state[None, :]})[0]
    move_probs = move_probs.reshape([go.N, go.N])
    
    coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    for move in sorted(coords, key=lambda c: move_probs[c], reverse=True):
      if go.is_eyeish(pos_cur.board, move):
        continue
      try:
        pos_next = pos_cur.play_move(move, mutate=False)
      except go.IllegalMove:
        continue
      else:
        return [pos_next, state, move[0]*go.N+move[1], 0]
    return None

  def sample_path(self, num_episodes = None):
    paths = []
    t = 0
    while t < self.batch_size:
      states, actions, rewards = [], [], []
      pos_cur = go.Position()
      while True:
        item= self.step(pos_cur)
        if item == None:
          score = pos_cur.score()
          if score == 0:
            rewards[-1]=0
            rewards[-2]=0
          elif (score>0 and pos_cur.to_play == go.WHITE) or (score < 0 and pos_cur.to_play == go.BLACK):
            rewards[-1]=1
            rewards[-2]=-1
          else:
            rewards[-1]=-1
            rewards[-2]=1
          break
        else:
          pos_cur = item[0]
          states.append(item[1])
          actions.append(item[2])
          rewards.append(item[3])        
          t += 1
          if t == self.batch_size:
            break
      path = {"observation" : np.array(states), 
                      "reward" : np.array(rewards), 
                      "action" : np.array(actions)}
      paths.append(path)  
    return paths
  
  
  def get_returns(self, paths):

    all_returns = []
    for path in paths:
      rewards = path["reward"]

      path_returns = np.zeros_like(rewards)
      cumulative = 0
      for t in reversed(range(len(rewards))):
        cumulative = cumulative * self.gamma + rewards[t]
        path_returns[t] = cumulative

      all_returns.append(path_returns)
    returns = np.concatenate(all_returns)
  
    return returns
  
  def calculate_advantage(self, returns, observations):
    adv = returns
    if self.use_baseline:
      baseline = self.sess.run(self.baseline, feed_dict={
                      self.observation_placeholder: observations})
      adv = adv - baseline
    if self.normalize_advantage:
      adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-6)
    return adv
  
  
  def update_baseline(self, returns, observations):
    self.sess.run(self.update_baseline_op, feed_dict={
                  self.observation_placeholder: observations,
                  self.baseline_target_placeholder: returns})

  def train(self, save_dir):
    for t in tqdm(range(self.num_batches)):
  
      # collect a minibatch of samples
      paths = self.sample_path() 
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)
      advantages = self.calculate_advantage(returns, observations)

      # run training operations
      if self.use_baseline:
        self.update_baseline(returns, observations)
      _, cost = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.observation_placeholder : observations, 
                    self.action_placeholder : actions, 
                    self.advantage_placeholder : advantages})
      self.stats.report(cost)
      if t % self.print_freq == 0:
        avg_cost, summary = self.stats.collect()
        global_step = self.sess.run(self.global_step)
        print("===: t: %d, step: %d; cost: %g" % (t, global_step, avg_cost))
        self.summary_writer.add_summary(summary, global_step)
      if t % self.checkpoint_freq == 0:
        self.save_variables(save_dir, t)

  def initialize(self, logdir, checkpoint=None):
    self.sess = tf.Session()
    self.saver_policy = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="PolicNetwork"))
    self.saver_value = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="ValueNetwork"))
    init = tf.global_variables_initializer()
    self.sess.run(init)
    if checkpoint is not None:
      self.saver_policy.restore(self.sess, checkpoint)
    self.initialize_logging(logdir)

  def save_variables(self, save_dir, t):
    print("Saving checkpoint %d" % t, file=sys.stderr)
    self.saver_policy.save(self.sess, os.path.join(save_dir, "batch_"+str(t)+"_policy.ckpt"))
    self.saver_value.save(self.sess, os.path.join(save_dir, "batch_"+str(t)+"_value.ckpt"))

  def run(self, save_dir, logdir, checkpoint=None):
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    if not os.path.exists(logdir):
      os.makedirs(logdir)
    self.initialize(logdir, checkpoint)
    self.train(save_dir)

  def initialize_logging(self, logdir):
    self.summary_writer = tf.summary.FileWriter(os.path.join(logdir, "train"), self.sess.graph)
    self.stats = StatisticsCollector()

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

if __name__ == '__main__':
    model = PG()
    model.run("checkpoint","log", "models/supervised/epoch_48.ckpt")
