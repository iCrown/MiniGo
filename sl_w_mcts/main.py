import argparse
import argh
import os
import random
import re
import sys

import gtp as gtp_lib

from policy import PolicyNetwork
from strategies import RandomPlayer, PolicyNetworkBestMovePlayer, PolicyNetworkRandomMovePlayer, MCTS
from load_data_sets import DataSet, parse_data_sets


def gtp(strategy, read_file=None):
    print("====gtp====")
    n = PolicyNetwork()
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

def preprocess(dataset_root, processed_dir="processed_data", desired_test_size=10**5):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
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

def train(processed_dir, save_dir, logdir, read_file=None, epochs=50, checkpoint_freq=2):
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

parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, preprocess, train])

if __name__ == '__main__':
    argh.dispatch(parser)
