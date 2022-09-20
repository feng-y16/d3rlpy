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


class D4RLDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, device):
        super(D4RLDataset, self).__init__()
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return min(len(self.dataset.observations) - 1, 10000)

    def __getitem__(self, item):
        return item

    def collate_fn(self, items):
        items = np.array(items)
        return torch.as_tensor(self.dataset.observations[items], dtype=torch.float32, device=self.device), \
               torch.as_tensor(self.dataset.actions[items], dtype=torch.float32, device=self.device), \
               torch.as_tensor(self.dataset.rewards[items], dtype=torch.float32, device=self.device), \
               torch.as_tensor(self.dataset.observations[items + 1], dtype=torch.float32, device=self.device), \
               torch.as_tensor(self.dataset.terminals[items], dtype=torch.float32, device=self.device)


def main(args):
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    assert not dataset.is_action_discrete()
    observation_space = spaces.Box(np.array([-np.finfo(np.float32).max] * dataset.get_observation_shape()[0],
                                            dtype=np.float32),
                                   np.array([np.finfo(np.float32).max] * dataset.get_observation_shape()[0],
                                            dtype=np.float32))
    action_space = spaces.Box(np.array([-1] * dataset.get_action_size(),
                                            dtype=np.float32),
                              np.array([1] * dataset.get_action_size(),
                                            dtype=np.float32))
    # prepare algorithm
    noda = NODA(dataset, observation_space, action_space, device=args.device).to(args.device)
    sac = d3rlpy.algos.SAC()
    pytorch_dataset = D4RLDataset(dataset, device=args.device)
    train_dataloader = torch.utils.data.DataLoader(pytorch_dataset, 256, collate_fn=pytorch_dataset.collate_fn,
                                                   shuffle=True, pin_memory=True, drop_last=True)
    # train offline
    noda.fit(train_dataloader, epochs=10)
    # train online
    sac.fit_online(noda, n_steps=200000)
    # ready to control
    buffer = sac.collect(gym.make("Hopper-v4"), n_steps=10001)
    reward = 0
    for i in range(10000):
        reward += buffer.transitions._buffer[i].reward
    reward /= 10000
    print(reward)


if __name__ == '__main__':
    main(parse_args())
