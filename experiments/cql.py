import d3rlpy
import pdb
import argparse
import torch
import numpy as np
import gym
from gym import spaces
from network import NODA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_known_args()[0]


def clip_dataset(dataset, length=10000):
    dataset._observations = dataset._observations[:length]
    dataset._actions = dataset._actions[:length]
    dataset._rewards = dataset._rewards[:length]
    dataset._terminals = dataset._terminals[:length]
    if dataset._episode_terminals is not None:
        dataset._episode_terminals = dataset._episode_terminals[:length]
    dataset.build_episodes()
    return dataset


def main(args):
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    assert not dataset.is_action_discrete()
    dataset = clip_dataset(dataset)
    # prepare algorithm
    cql = d3rlpy.algos.CQL(use_gpu=True)

    # train
    cql.fit(
        dataset,
        eval_episodes=dataset,
        n_epochs=100,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer,
        },
        save_metrics=False,
    )
    # ready to control
    buffer = cql.collect(env, n_steps=10001)
    reward = 0
    for i in range(10000):
        reward += buffer.transitions._buffer[i].reward
    reward /= 10000
    print(reward)


if __name__ == '__main__':
    main(parse_args())
