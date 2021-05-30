#!/usr/bin/env python3
import os
import argparse
import time
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


####### Define a Neural Network Model
## Input: features of a link (length, zone crossing, other global features: number of stops etc)
## Output: Abstract Distance of a link
## trainable parameters: a fully-connect neural net and lambda
## Loss: See Overleaf

#########


def compute_tsp_seq_for_route(data, theta, lamb):
    travel_times, link_features, route_features, time_constraints, \
        stop_ids, travel_time_dict, label = data

    demo_stop_ids = [stop_ids[j] for j in label]
    demo_stop_ids.append(demo_stop_ids[0])

    ##### Replace the following part with Neural Network -- forward pass
    num_link_features, num_stops, _ = link_features.shape
    temp = link_features.reshape(num_link_features, -1).T
    temp = temp.dot(theta)
    objective_matrix = temp.reshape(num_stops, num_stops) + 1 * travel_times

    # objective_matrix = forward_pass(travel_times, link_feature, route_features)

    ###########

    pred_seq = constrained_tsp.constrained_tsp(
        objective_matrix, travel_times, time_constraints, depot=label[0], lamb=int(lamb))

    ####### Remove this part
    ## time_cost and feature_cost not needed in computing grad
    ## since we have replaced it with a neural net
    pred_time_cost, pred_feature_cost = compute_link_cost_seq(travel_times, link_features, pred_seq)
    demo_time_cost, demo_feature_cost = compute_link_cost_seq(travel_times, link_features, label)
    ########

    pred_tv = compute_time_violation_seq(travel_times, time_constraints, pred_seq)
    demo_tv = compute_time_violation_seq(travel_times, time_constraints, label)

    pred_stop_ids = [stop_ids[j] for j in pred_seq]
    pred_stop_ids.append(pred_stop_ids[0])

    seq_score = score(demo_stop_ids, pred_stop_ids, travel_time_dict)

    return pred_seq, demo_time_cost, demo_feature_cost, demo_tv, \
        pred_time_cost, pred_feature_cost, pred_tv, seq_score


def compute_tsp_seq_for_a_batch(batch_data, theta, lamb):
    pool = Pool(processes=8)
    tsp_func = partial(compute_tsp_seq_for_route, theta=theta, lamb=lamb)
    batch_output = pool.map(tsp_func, batch_data)
    pool.close()
    return batch_output


def get_param(batch_demo_time_cost, batch_demo_feature_cost, batch_demo_tv,
              batch_pred_time_cost, batch_pred_feature_cost, batch_pred_tv,):
    X1 = batch_demo_feature_cost - batch_pred_feature_cost
    X2 = np.expand_dims(batch_demo_tv - batch_pred_tv, axis=1)
    X = np.concatenate((X1, X2), axis=1)
    y = - (batch_demo_time_cost - batch_pred_time_cost)
    params = np.linalg.lstsq(X, y, rcond=None)
    theta = params[0][:-1]
    lamb = params[0][-1]
    return theta, lamb


def fit(model, dataloader, writer, config):
    theta = np.array([50.0])
    lamb = 10.0
    lr = config.learning_rate
    clock = 0

    # loop over the dataset multiple times
    for epoch_idx in range(config.num_train_epochs):
        epoch_score = 0
        epoch_loss = 0

        clock += 1

        print("Theta: {}, Lambda: {}".format(theta, lamb))

        batch_data = [None] * config.batch_size
        for idx, data in enumerate(dataloader):
            batch_data[idx % config.batch_size] = data
            if idx % config.batch_size == config.batch_size - 1:
                start_time = time.time()
                batch_output = compute_tsp_seq_for_a_batch(batch_data, theta, lamb)
                res = list(zip(*batch_output))
                batch_demo_time_cost = np.array(res[1])
                batch_demo_feature_cost = np.array(res[2])
                batch_demo_tv = np.array(res[3])
                batch_pred_time_cost = np.array(res[4])
                batch_pred_feature_cost = np.array(res[5])
                batch_pred_tv = np.array(res[6])
                batch_seq_score = np.array(res[7])

                # compute gradient
                grad_lamb_batch = batch_demo_tv - batch_pred_tv
                grad_theta_batch = batch_demo_feature_cost - batch_pred_feature_cost

                demo_cost = 1 * batch_demo_time_cost + theta.dot(batch_demo_feature_cost.T)
                pred_cost = 1 * batch_pred_time_cost + theta.dot(batch_pred_feature_cost.T)

                loss = max((np.mean(demo_cost) + lamb * np.mean(batch_demo_tv)) -
                           (np.mean(pred_cost) + lamb * np.mean(batch_pred_tv)), 0)

                # update theta and lambda
                # r = lr / (1 + clock * 0.0005)
                # lamb -= np.mean(grad_lamb_batch) * r
                # theta -= np.mean(grad_theta_batch) * r

            ##### Replace the following part with Neural Net -- backward pass
                # Init parameters by finding optimal paramters
                theta, lamb = get_param(batch_demo_time_cost,
                                        batch_demo_feature_cost,
                                        batch_demo_tv,
                                        batch_pred_time_cost,
                                        batch_pred_feature_cost,
                                        batch_pred_tv)

                # we need be very careful about loss computation
                # loss.backward()

            #####

                mean_score = np.mean(batch_seq_score)
                epoch_loss += loss
                epoch_score += mean_score

                writer.add_scalar('Train/batch_route_loss', loss, epoch_idx * len(dataloader) + idx)
                writer.add_scalar('Train/batch_route_score', mean_score, epoch_idx * len(dataloader) + idx)

                print("Epoch: {}, Step: {}, Loss: {:01f}, Score: {:01f}, Time: {}, Theta: {}, Lambda: {}".
                      format(epoch_idx, int(idx / config.batch_size), loss, mean_score, time.time() - start_time, theta, lamb))

        mean_epoch_loss = (epoch_loss * config.batch_size) / (len(dataloader))
        mean_epoch_score = (epoch_score * config.batch_size) / (len(dataloader))

        print("Epoch Loss: {}, Epoch Score: {}".format(mean_epoch_loss, mean_epoch_score))

        writer.add_scalar('Train/loss', mean_epoch_loss, epoch_idx)
        writer.add_scalar('Train/score', mean_epoch_score, epoch_idx)


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    data = IRLDataset(config.data)
    writer = SummaryWriter(
        logdir=config.tensorboard_dir +
        '/{}_{}_Model'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            config.name))
    model = LinearModel(size=2)
    fit(model, data.x, writer, config)
    print('Finished Training')


if __name__ == '__main__':
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
