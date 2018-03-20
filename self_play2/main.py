import argparse
import argh
import os
import random
import re
import sys
import go
import gtp as gtp_lib
import tensorflow as tf
import numpy as np

from policy import PolicyNetwork
from strategies import RandomPlayer, PolicyNetworkBestMovePlayer, PolicyNetworkRandomMovePlayer, MCTS
from load_data_sets import DataSet, parse_data_sets, parse_selfplay_sets
import features


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


def make_move_from_prob(pos_cur, move_probs):
    coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    coords = sorted(coords, key=lambda c: move_probs[c], reverse=True)
    
    for move in coords:
      if go.is_eyeish(pos_cur.board, move):
        continue
      try:
        pos_next = pos_cur.play_move(move, mutate=False)
      except go.IllegalMove:
        continue
      else:
        return move
    return None

def random_move_from_pos(pos_cur):
    coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    random.shuffle(coords)
    
    for move in coords:
      if go.is_eyeish(pos_cur.board, move):
        continue
      try:
        pos_next = pos_cur.play_move(move, mutate=False)
      except go.IllegalMove:
        continue
      else:
        return move
    return None


def shuffles(observations, actions, rewards):
    perm = np.arange(len(observations))
    np.random.shuffle(perm)
    observations = observations[perm]
    actions = actions[perm]
    rewards = rewards[perm]
    return observations, actions, rewards

def clean_path(path):
    observation = path['observation']
    action = path['action']
    reward = path['reward']

    # remove none action
    idx = []
    for i in range(len(action)):
        if action[i]:
            idx.append(i)
    action = np.take(action, idx)
    observation = np.take(observation, idx, axis = 0)
    reward = np.take(reward, idx)
    path = {"observation" : observation, 
              "reward" : reward, 
              "action" : action}
    return path

def gtp(strategy, read_file, scope):
    print("====gtp====")
    n = PolicyNetwork(scope)
    if strategy == 'random':
        instance = RandomPlayer()
    elif strategy == 'policy':
        instance = PolicyNetworkBestMovePlayer(n, read_file)
    elif strategy == 'randompolicy':
        instance = PolicyNetworkRandomMovePlayer(n, read_file)
    elif strategy == 'mcts':
        instance = MCTS(n, read_file)
    else:
        sys.stderr.write("Unknown strategy")
        sys.exit()
    gtp_engine = gtp_lib.Engine(instance)
    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not gtp_engine.disconnect:
        inpt = input()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = gtp_engine.send(cmd)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()

def preprocess(dataset_root, processed_dir="processed_data", desired_test_size=0.05, selfplay = True):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if selfplay:
        # convert from saved gz file from selfplay to train/test data
        test_dataset, train_dataset = parse_selfplay_sets(dataset_root, processed_dir, desired_test_size)
        print("=====Writing test chunk=====")
        test_filename = os.path.join(processed_dir, "test.chunk.gz")
        test_dataset.write(test_filename)

        print("=====Writing training chunk=====")
        train_filename = os.path.join(processed_dir, "train.chunk.gz")
        train_dataset.write(train_filename)

    else:
        # convert from sgf game file to train/test data
        test_chunk, train_chunk = parse_data_sets(dataset_root, desired_test_size)
        print("=====Test # %s, Train # %s.=====" % (len(test_chunk), len(train_chunk)))

        print("=====Writing test chunk=====")
        test_dataset = DataSet.from_positions_w_context(test_chunk, is_test=True)
        test_filename = os.path.join(processed_dir, "test.chunk.gz")
        test_dataset.write(test_filename)

        print("=====Writing training chunk=====")
        train_dataset = DataSet.from_positions_w_context(train_chunk, is_test=True)
        train_filename = os.path.join(processed_dir, "train.chunk.gz")
        train_dataset.write(train_filename)


def train(processed_dir='processed_data', save_dir='model', logdir='logs', read_file=None, epochs=50, checkpoint_freq=2):
    test_dataset = DataSet.read(os.path.join(processed_dir, "test.chunk.gz"))
    train_dataset = DataSet.read(os.path.join(processed_dir, "train.chunk.gz"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("=====Network initilization=====")
    net = PolicyNetwork(logdir=logdir, read_file=read_file)
    print("=====Start training...=====")
    for i in range(epochs):
        net.train(train_dataset, test_dataset)
        if i % checkpoint_freq == 0:
            net.save_variables(os.path.join(save_dir, "epoch_"+str(i)+".ckpt"))


def selfplay(train_net, player_net):
    states, actions, rewards = [], [], []
    position = go.Position()
    move = (random.randint(2,7), random.randint(2,7))
    position = position.play_move(move)
    current = 1 if random.random() > 0.5 else -1
    step = 0
    while True:
        if position.is_game_over():
            print ("current: "+str(current)+"; to_play: "+str(position.to_play) + "; over")
            score = position.score()
            print (score)
            if (score>0 and position.to_play == go.WHITE) or (score < 0 and position.to_play == go.BLACK):
                rewards[-1]=1
                rewards[-2]=-1
            else:
                rewards[-1]=-1
                rewards[-2]=1
            break

        if step < 5 and np.random.rand() < 0.5:
                move = random_move_from_pos(position)
        else:
            if current == 1:
                move_probs = train_net.run(position)
                move = make_move_from_prob(position, move_probs)
            if current == -1:
                move_probs = player_net.run(position)
                move = make_move_from_prob(position, move_probs)
        current *= -1

        if move == gtp_lib.RESIGN or position.caps[0] + 25 < position.caps[1]:
            print ("current: "+str(current)+"; to_play: "+str(position.to_play) + "; resign")
            score = position.score()
            print (score)
            if (score>0 and position.to_play == go.WHITE) or (score < 0 and position.to_play == go.BLACK):
                rewards[-1]=1
                rewards[-2]=-1
            else:
                rewards[-1]=-1
                rewards[-2]=1
            break

        state = features.extract_features(position)
        states.append(state)
        if move:
            actions.append(move[0]*go.N+move[1])
        else:
            actions.append(None)
        rewards.append(0)

        position = position.play_move(move)
        step +=1

    # get reward for each state
    path_return = np.zeros_like(rewards)
    path_return[-1] = rewards[-1]
    path_return[-2] = rewards[-2]
    for t in reversed(range(len(rewards))):
        if t-2 >=0:
            path_return[t-2] = path_return[t]

    path = {"observation" : np.array(states), 
              "reward" : np.array(path_return), 
              "action" : np.array(actions)}
    return clean_path(path)


def selfplay_mcts(train_net, player_net):
    states, actions, rewards = [], [], []
    position = go.Position()
    train_player = MCTS(train_net)
    train_player.root.inject_noise()
    current = 1 if random.random() > 0.5 else -1
    first = False if current == 1 else True
    step = 0
    while True:
        if position.is_game_over():
            print ("current: "+str(current)+"; to_play: "+str(position.to_play) + "; over")
            score = position.score()
            print (score)
            if (score>0 and position.to_play == go.WHITE) or (score < 0 and position.to_play == go.BLACK):
                rewards[-1]=1
                rewards[-2]=-1
            else:
                rewards[-1]=-1
                rewards[-2]=1
            break


        if step < 5 and np.random.rand() < 0.5:
            move = random_move_from_pos(position)
        else:
            if current == 1:
                move = train_player.suggest_move(position)
            if current == -1:
                move_probs = player_net.run(position)
                move = make_move_from_prob(position, move_probs)
        current *= -1

        if move == gtp_lib.RESIGN:
            print ("current: "+str(current)+"; to_play: "+str(position.to_play) + "; resign")
            score = position.score()
            print (score)
            if (score>0 and position.to_play == go.WHITE) or (score < 0 and position.to_play == go.BLACK):
                rewards[-1]=1
                rewards[-2]=-1
            else:
                rewards[-1]=-1
                rewards[-2]=1
            break

        state = features.extract_features(position)
        states.append(state)
        if move:
            actions.append(move[0]*go.N+move[1])
        else:
            actions.append(None)
        rewards.append(0)

        position = position.play_move(move)

        print (position)
        step +=1

    # get reward for each state
    path_return = np.zeros_like(rewards)
    path_return[-1] = rewards[-1]
    path_return[-2] = rewards[-2]
    for t in reversed(range(len(rewards))):
        if t-2 >=0:
            path_return[t-2] = path_return[t]

    path = {"observation" : np.array(states), 
              "reward" : np.array(path_return, dtype = np.float32), 
              "action" : np.array(actions, dtype = np.int8)}

    print (path)
    return clean_path(path)



def reinforce(read_file = 'model/sl/epoch_48.ckpt', save_dir = 'model/rl/', logdir = 'log', num_iter = 2000, games_per_iter = 10, mcts = 1, save_freq = 100):
    train_net = PolicyNetwork(logdir = logdir, scope = "PolicNetwork")
    train_net.initialize_variables(read_file)
    
    for i in range(num_iter):
        if i % save_freq == 0 and i > 0:
            train_net.save_variables(os.path.join('model/sl/',  "polic"+str(i)+".ckpt"))
            save_trained_policy(int(i/save_freq), save_dir)
            games_per_iter *= max(1, int(i**(10**-1)))

        players = [player for player in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, player))]
        np.random.shuffle(players)
        print ("random choosing an opponent player from a pool of "+str(len(players)))
        
        paths = []
        for game in range(games_per_iter):
            t = players[game%len(players)]

            # initialize new graph for opponent player
            g2 = tf.Graph()
            with g2.as_default():
                player_net = PolicyNetwork(scope = "PlayerNetwork")
                player_net.initialize_variables(os.path.join(save_dir, str(t), "player"+str(t)+".ckpt"))
                print ("opponent player "+str(t)+", in game "+str(game)+", in iter "+str(i))

            if random.random() > mcts:
                print ("============= mcts selfplay ===============")
                path = selfplay_mcts(train_net, player_net)
            else:
                print ("============= selfplay ===============")
                path = selfplay(train_net, player_net)

            paths.append(path)

        observations = np.concatenate([path["observation"].astype(np.float32) for path in paths])
        actions = np.concatenate([path["action"].astype(np.int8) for path in paths])
        rewards = np.concatenate([path["reward"].astype(np.float32) for path in paths])
        #observations, actions, rewards = shuffles(observations, actions, rewards)
        print ("training")
        train_net.train(observations, actions, rewards)
        pos_test = go.Position()
        print (train_net.run(pos_test))
        print ("training done")
    train_net.save_variables(os.path.join('model/sl/',  "polic_final.ckpt"))


parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, preprocess, train, selfplay, reinforce])

if __name__ == '__main__':
    argh.dispatch(parser)
