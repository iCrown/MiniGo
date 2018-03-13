import argparse
import argh
import os
import random
import re
import sys

from policy import PolicyNetwork
from load_data_sets import DataSet, parse_data_sets


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
argh.add_commands(parser, [preprocess, train])

if __name__ == '__main__':
    argh.dispatch(parser)
