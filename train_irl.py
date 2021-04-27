import os
import argparse
import torch
import datetime
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
import numpy as np
from easydict import EasyDict as edict
import yaml
import pprint

from models.models import LinearModel
from dataloaders.irl_dataset import IRLDataset
from training_utils.arg_utils import get_args, setup_training_output
from tsp_solvers import tsp, constrained_tsp
from IPython import embed


#%%
null_callback = lambda *args, **kwargs: None


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


def fit(model, dataloader, writer, config):
    theta = np.array([100])
    lamb = 10
    learning_rate = 0.001
    clock = 0
    for epoch in range(config.num_train_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader):
            clock += 1
            route_loss = []
            travel_times, link_features, route_features, time_constraints, label = data

            for grad_idx in range(config.num_grad_steps):

                # travel_time_matrix = model(link_features, route_features)
                # travel_time_matrix = travel_time_matrix.numpy()

                num_link_features, num_stops, _ = link_features.shape
                temp = link_features.reshape(num_link_features, -1).T
                temp = temp.dot(theta)
                objective_matrix = temp.reshape(num_stops, num_stops) + 1 * travel_times

                print(objective_matrix.shape, travel_times.shape, len(time_constraints), len(label))

                pred_seq = constrained_tsp.constrained_tsp(
                    objective_matrix, travel_times, time_constraints,
                    depot=label[0], lamb=lamb)

                print(i, np.array(pred_seq))

                # pred_seq = tsp.compute_tsp_solution(distance_matrix=travel_time_matrix, depot=label[0])

                # tsp_solution = constrained_tsp.constrained_tsp(travel_time_matrix=travel_time_matrix.detach().numpy(),
                #                                time_window_list=time_constraints,
                #                                depot=label[0])

                demo_time_cost, demo_feature_cost = compute_link_cost_seq(
                    travel_times, link_features, label)
                pred_time_cost, pred_feature_cost = compute_link_cost_seq(
                    travel_times, link_features, pred_seq)

                demo_cost = 1 * demo_time_cost + theta.dot(demo_feature_cost)
                pred_cost = 1 * pred_time_cost + theta.dot(pred_feature_cost)

                demo_tv = compute_time_violation_seq(
                    travel_times, time_constraints, label)
                pred_tv = compute_time_violation_seq(
                    travel_times, time_constraints, pred_seq)

                # compute gradient
                grad_lamb = demo_tv - pred_tv
                grad_theta = demo_feature_cost - pred_feature_cost

                # update theta and lambda
                r = learning_rate / (1 + clock * 0.0005)
                lamb -= grad_lamb * r
                theta -= grad_theta * r

                loss = (demo_cost + lamb * demo_tv) - (pred_cost + lamb * pred_tv)
                embed()

                route_loss.append(loss)
                # writer.add_scalar('Train/route_{}_loss'.format(i), loss, grad_idx)


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    data = IRLDataset(config.data)
    writer = SummaryWriter(logdir=config.tensorboard_dir + '/{}_{}_Model'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config.name))
    model = LinearModel(size=2)
    fit(model, data.x, writer, config)
    print('Finished Training')


if __name__ == '__main__':
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
