from policy import PolicyNetwork
import tensorflow as tf
import go
import os
import numpy as np


def save_trained_policy(t, save_dir):
	with tf.Graph().as_default(), tf.Session().as_default() as sess:
		policy_vars = tf.contrib.framework.list_variables('model/sl/')
		new_vars = []
		for name, shape in policy_vars:
			v = tf.contrib.framework.load_variable('model/sl/', name)
			new_vars.append(tf.Variable(v, name=name.replace('PolicNetwork', 'PlayerNetwork')))
		saver = tf.train.Saver(new_vars)
		sess.run(tf.global_variables_initializer())
		saver.save(sess, os.path.join(save_dir, str(t), 'player'+str(t)+'.ckpt'))


g1 = tf.Graph()
with g1.as_default():
    train_net = PolicyNetwork(scope = "PolicNetwork")
    train_net.initialize_variables('model/sl/epoch_48.ckpt')
    
    
pos = go.Position()
train_net.run(pos)

g2 = tf.Graph()
with g2.as_default():
	player_net = PolicyNetwork(scope = "PlayerNetwork")
	player_net.initialize_variables('model/rl/2/player2.ckpt')
pos = go.Position()
player_net.run(pos)


save_trained_policy(1, 'model/rl')

print ("===========load new model=================")
g2 = tf.Graph()
with g2.as_default():
	player_net = PolicyNetwork(scope = "PlayerNetwork")
	player_net.initialize_variables('model/rl/5/player5.ckpt')
move_probs10 = player_net.run(pos)
assert np.array_equal(move_probs10, move_probs0)
print ("succefully loaded")


with tf.Session().as_default() as sess:
	train_net.save_variables('checkpoint1/save/policy.ckpt')
	





