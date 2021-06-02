import os
import argparse
import time
from numpy.lib.function_base import bartlett
import torch
import datetime
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
import numpy as np
from easydict import EasyDict as edict
import yaml
import pprint
from multiprocessing import Pool
from functools import partial
import tqdm

from models.models import LinearModel
from dataloaders.irl_dataset import IRLDataset
from training_utils.arg_utils import get_args, setup_training_output
from tsp_solvers import tsp, constrained_tsp
from eval_utils.score import score
from IPython import embed


def compute_link_cost_seq(time_matrix, link_features, seq):
    time_cost = 0
    feature_cost = np.zeros(link_features.shape[0])
    for idx in range(1, len(seq)):
        time_cost += time_matrix[seq[idx-1], seq[idx]]
        feature_cost += link_features[:, seq[idx-1], seq[idx]]
    time_cost += time_matrix[seq[-1]][seq[0]]
    feature_cost += link_features[:, seq[-1], seq[0]]
    return time_cost, feature_cost


def compute_time_violation_seq(time_matrix, time_constraints, seq):
    time = 0.0
    violation_down, violation_up = 0, 0
    for idx in range(1, len(seq)):
        time += time_matrix[seq[idx-1], seq[idx]]
        violation_up += max(0, time - time_constraints[idx][1])
        violation_down += max(0, time_constraints[idx][0] - time)
    return violation_up + violation_down


def compute_tsp_seq_for_route(data, theta, lamb):
    travel_times, link_features, route_features, time_constraints, \
        stop_ids, travel_time_dict, label = data
    
    demo_stop_ids = [stop_ids[j] for j in label]
    demo_stop_ids.append(demo_stop_ids[0])
    
    num_link_features, num_stops, _ = link_features.shape
    temp = link_features.reshape(num_link_features, -1).T
    temp = temp.dot(theta)
    objective_matrix = temp.reshape(num_stops, num_stops) + 1 * travel_times
    
    pred_seq = constrained_tsp.constrained_tsp(
        objective_matrix, travel_times, time_constraints, depot=label[0], lamb=int(lamb))

    tsp_seq = constrained_tsp.constrained_tsp(
        travel_times, travel_times, time_constraints, depot=label[0], lamb=int(lamb))
    
    pred_stop_ids = [stop_ids[j] for j in pred_seq]
    pred_stop_ids.append(pred_stop_ids[0])
    pred_seq_score = score(demo_stop_ids, pred_stop_ids, travel_time_dict)

    tsp_stop_ids = [stop_ids[j] for j in tsp_seq]
    tsp_stop_ids.append(tsp_stop_ids[0])
    tsp_seq_score = score(demo_stop_ids, tsp_stop_ids, travel_time_dict)
    
    return pred_seq_score, tsp_seq_score


def compute_tsp_seq_for_a_batch(batch_data, theta, lamb):
    pool = Pool(processes=8)
    tsp_func = partial(compute_tsp_seq_for_route, theta=theta, lamb=lamb)
    # batch_output = pool.map(tsp_func, batch_data)
    batch_output = list(tqdm.tqdm(pool.imap(tsp_func, batch_data), total=len(batch_data)))
    pool.close()
    return batch_output


def eval(dataloader, config):
    theta = config.theta
    lamb = config.lamb
    print("Theta: {}, Lambda: {}".format(theta, lamb))

    batch_output1 = compute_tsp_seq_for_a_batch(dataloader[:500], theta, lamb)
    res = list(zip(*batch_output1))
    pred_score = np.sum(res[0])
    tsp_score = np.sum(res[1])
    print("IRL Score: {:01f}, Raw TSP Score: {:01f}".format(pred_score / 500.0, tsp_score / 500.0))


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    data = IRLDataset(config.data)
    eval(data.x, config)
    print('Finished Eval')


if __name__ == '__main__':
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)