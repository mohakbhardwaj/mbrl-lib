import copy
import csv
import json
import math
import random
import string
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn as nn
from psutil import disk_io_counters

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torchify(x):
    x = torch.tensor(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), DEFAULT_DEVICE  # dataset[k].device
    for v in dataset.values():
        assert len(v) == n, "Dataset values must have same length"
    indices = np.random.randint(low=0, high=n, size=(batch_size,))  # , device=device)
    return {k: torchify(v[indices]) for k, v in dataset.items()}


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def traj_to_tuple_data(traj_data, ignores=("metadata",)):
    """Concatenate a list of trajectory dicts to a dict of np.arrays of the same length."""
    tuple_data = dict()
    for k in traj_data[0].keys():
        if not any([ig in k for ig in ignores]):
            tuple_data[k] = np.concatenate([traj[k] for traj in traj_data])
    return tuple_data


def tuple_to_traj_data(tuple_data, ignores=("metadata",)):
    """Split a tuple_data dict in d4rl format to list of trajectory dicts."""
    tuple_data["timeouts"][-1] = not tuple_data["terminals"][-1]
    ends = (tuple_data["terminals"] + tuple_data["timeouts"]) > 0
    ends[-1] = False  # don't need to split at the end
    inds = np.arange(len(ends))[ends] + 1
    tmp_data = dict()
    for k, v in tuple_data.items():
        if not any([ig in k for ig in ignores]):
            tmp_data[k] = np.split(v, inds)
    traj_data = [
        dict(zip(tmp_data, t)) for t in zip(*tmp_data.values())
    ]  # convert to list of dict
    return traj_data


def traj_data_to_qlearning_data(traj_data, ignores=("metadata",)):
    """Convert a list of trajectory dicts into d4rl qlearning data format."""
    traj_data = copy.deepcopy(traj_data)
    for traj in traj_data:
        # process 'observations'
        if traj["terminals"][-1] > 0:
            traj["observations"] = np.append(
                traj["observations"], traj["observations"][-1:], axis=0
            )  # duplicate
        else:  # ends because of timeout
            for k, v in traj.items():
                if k != "observations":
                    traj[k] = v[:-1]
        # At this point, traj['observations'] should have one more element than the others.
        traj["next_observations"] = traj["observations"][1:]
        traj["observations"] = traj["observations"][:-1]
        lens = [len(v) for k, v in traj.items()]
        assert all([lens[0] == l for l in lens[1:]])

    return traj_to_tuple_data(traj_data, ignores=ignores)


# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), DEFAULT_DEVICE  # dataset[k].device
    for v in dataset.values():
        assert len(v) == n, "Dataset values must have same length"
    indices = np.random.randint(low=0, high=n, size=(batch_size,))  # , device=device)
    return {k: torchify(v[indices]) for k, v in dataset.items()}


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def traj_to_tuple_data(traj_data, ignores=("metadata",)):
    """Concatenate a list of trajectory dicts to a dict of np.arrays of the same length."""
    tuple_data = dict()
    for k in traj_data[0].keys():
        if not any([ig in k for ig in ignores]):
            tuple_data[k] = np.concatenate([traj[k] for traj in traj_data])
    return tuple_data


def tuple_to_traj_data(tuple_data, ignores=("metadata",)):
    """Split a tuple_data dict in d4rl format to list of trajectory dicts."""
    tuple_data["timeouts"][-1] = not tuple_data["terminals"][-1]
    ends = (tuple_data["terminals"] + tuple_data["timeouts"]) > 0
    ends[-1] = False  # don't need to split at the end
    inds = np.arange(len(ends))[ends] + 1
    tmp_data = dict()
    for k, v in tuple_data.items():
        if not any([ig in k for ig in ignores]):
            tmp_data[k] = np.split(v, inds)
    traj_data = [
        dict(zip(tmp_data, t)) for t in zip(*tmp_data.values())
    ]  # convert to list of dict
    return traj_data


def traj_data_to_qlearning_data(traj_data, ignores=("metadata",)):
    """Convert a list of trajectory dicts into d4rl qlearning data format."""
    traj_data = copy.deepcopy(traj_data)
    for traj in traj_data:
        # process 'observations'
        if traj["terminals"][-1] > 0:
            traj["observations"] = np.append(
                traj["observations"], traj["observations"][-1:], axis=0
            )  # duplicate
        else:  # ends because of timeout
            for k, v in traj.items():
                if k != "observations":
                    traj[k] = v[:-1]
        # At this point, traj['observations'] should have one more element than the others.
        traj["next_observations"] = traj["observations"][1:]
        traj["observations"] = traj["observations"][:-1]
        lens = [len(v) for k, v in traj.items()]
        assert all([lens[0] == l for l in lens[1:]])

    return traj_to_tuple_data(traj_data, ignores=ignores)


def evaluate_policy(env, policy, max_episode_steps, deterministic=True, discount=0.99):
    obs = env.reset()
    total_reward = 0.0
    discount_total_reward = 0.0
    for i in range(max_episode_steps):
        with torch.no_grad():
            action = (
                policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
            )
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        discount_total_reward += reward * discount**i
        if done:
            break
        else:
            obs = next_obs
    return [total_reward, discount_total_reward]


def discount_cumsum(x, discount):
    """Discounted cumulative sum.
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    Here, we have y[t] - discount*y[t+1] = x[t]
    or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    Args:
        x (np.ndarrary): Input.
        discount (float): Discount factor.
    Returns:
        np.ndarrary: Discounted cumulative sum.
    """
    import scipy.signal as signal

    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=-1)[::-1]


# (Deprecated) Adding future reward
def add_future_reward(dataset, discount_factor):
    tem = dataset[
        "terminals"
    ]  # This is problematic: There should be some cutoff trajectory add or time out.
    #
    # Sum the ovbserved reward until stop, at timeout add the target network prediciton. (with discounts.)
    # Learn a network predicing that
    #
    rew = dataset["rewards"]
    # dataset['future_rewards'] = rew
    # return dataset
    reward_cuts = np.split(rew, np.arange(len(tem))[tem] + 1)
    future_rewards = []
    for rw in reward_cuts:
        return_to_go = discount_cumsum(rw, discount_factor)[1:]
        return_to_go = np.append(return_to_go, 0)
        future_rewards.append(return_to_go)
    future_rewards = np.concatenate(future_rewards)
    assert len(future_rewards) == len(rew)
    dataset["future_rewards"] = future_rewards

    # reward_cut = np.split(rew, np.arange(len(tem))[tem]+1)
    # future_rewards = []
    # future_rewards = []
    # for i in range(tem.sum()+1):
    #     for j in range(len(reward_cut[i])-1):
    #         #len_future = len(reward_cut[i]) - j - 1
    #         len_future = min(len(reward_cut[i]) - j - 1, 100000)
    #         discounts = discount_factor**(np.arange(len_future))
    #         future_rewards += [(reward_cut[i][(j+1):(len_future+j+1)]*discounts).sum()]
    #     future_rewards += [0]
    # assert len(future_rewards) == len(rew)
    # dataset['future_rewards'] = np.array(future_rewards)

    return dataset


def simple_lambda(future_rewards, round_threshold=0.1):
    re = []
    for i in range(len(future_rewards)):
        re += [(future_rewards <= future_rewards[i]).float().mean()]
    re = torch.stack(re)
    re[re > 1 - round_threshold] = 1
    re[re < round_threshold] = 0
    return re