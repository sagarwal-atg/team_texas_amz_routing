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
from eval_utils.score import score
from IPython import embed


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
    theta = np.array([100.0])
    lamb = 100.0
    learning_rate = 0.001
    clock = 0

    # loop over the dataset multiple times
    for epoch_idx in range(config.num_train_epochs):
        avg_epoch_score = 0

        all_route_scores = []
        
        for grad_idx in range(config.num_grad_steps):
            clock += 1
            
            route_loss = [0] * len(dataloader)
            route_score = [0] * len(dataloader)
            grad_lamb_batch = [0] * len(dataloader)
            grad_theta_batch = [0] * len(dataloader)
            best_score = [np.inf] * len(dataloader)

            for idx, data in enumerate(dataloader):
                travel_times, link_features, route_features, time_constraints, \
                    stop_ids, travel_time_dict, label = data
                
                demo_stop_ids = [stop_ids[j] for j in label ]
                demo_stop_ids.append(demo_stop_ids[0])

                num_link_features, num_stops, _ = link_features.shape
                temp = link_features.reshape(num_link_features, -1).T
                temp = temp.dot(theta)
                objective_matrix = temp.reshape(num_stops, num_stops) + 1 * travel_times

                pred_seq = constrained_tsp.constrained_tsp(
                    objective_matrix, travel_times, time_constraints,
                    depot=label[0], lamb=int(lamb))

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
                grad_lamb_batch[idx] = demo_tv - pred_tv
                grad_theta_batch[idx] =  demo_feature_cost - pred_feature_cost

                loss = max((demo_cost + lamb * demo_tv) - (pred_cost + lamb * pred_tv), 0)

                pred_stop_ids = [stop_ids[j] for j in pred_seq ]
                pred_stop_ids.append(pred_stop_ids[0])

                seq_score = score(demo_stop_ids, pred_stop_ids, travel_time_dict)

                best_score[idx] = seq_score if seq_score < best_score[idx] else best_score[idx]

                route_score[idx] = seq_score
                route_loss[idx] = loss

                # writer.add_scalar('Train/route_{}_loss'.format(i), loss, epoch_idx * config.num_grad_steps + grad_idx)
                writer.add_scalar('Scores/route_{}_score'.format(idx), seq_score, epoch_idx * config.num_grad_steps + grad_idx)

            avg_grad_lamb = sum(grad_lamb_batch) / len(grad_lamb_batch)
            avg_grad_theta = sum(grad_theta_batch) / len(grad_theta_batch)

            # update theta and lambda
            r = learning_rate / (1 + clock * 0.0005)
            lamb -= avg_grad_lamb * r
            theta -= avg_grad_theta * r

            avg_loss = sum(route_loss) / len(route_loss)
            avg_score = sum(route_score) / len(route_score)
            print("Epoch: {}, Grad Step: {}, Loss: {}, Score: {:02f}".format(epoch_idx, grad_idx, avg_loss, avg_score))
            
            writer.add_scalar('Train/avg_route_loss', avg_loss, epoch_idx * config.num_grad_steps + grad_idx)
            writer.add_scalar('Train/avg_route_score', avg_score, epoch_idx * config.num_grad_steps + grad_idx)
            
            all_route_scores.append(avg_score)
        
        avg_epoch_score = sum(all_route_scores) / len(all_route_scores)
        print("Epoch: {}, Avg Score: {}".format(epoch_idx, avg_epoch_score))
        writer.add_scalar('Train/avg_epoch_score', avg_epoch_score, epoch_idx)


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
