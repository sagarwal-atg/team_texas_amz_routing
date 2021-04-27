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



null_callback = lambda *args, **kwargs: None


def compute_cost_seq(matrix, seq):
    cost = 0.0
    for idx in range(1, len(seq)):
        cost += matrix[seq[idx-1], seq[idx]]
    return cost


def fit(model, dataloader, writer, config):
    for epoch in range(config.num_train_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(dataloader):
            route_loss = []
            link_features, route_features, time_constraints, label = data

            for grad_idx in range(config.num_grad_steps):

                travel_time_matrix = model(link_features, route_features)
                travel_time_matrix = travel_time_matrix.numpy()

                pred_seq = tsp.compute_tsp_solution(distance_matrix=travel_time_matrix, depot=label[0])

                # tsp_solution = constrained_tsp.constrained_tsp(travel_time_matrix=travel_time_matrix.detach().numpy(),
                #                                time_window_list=time_constraints,
                #                                depot=label[0])

                demo_cost = compute_cost_seq(travel_time_matrix, label)
                pred_cost = compute_cost_seq(travel_time_matrix, pred_seq)

                loss = demo_cost - pred_cost
                embed()

                route_loss.append(loss)
                writer.add_scalar('Train/route_{}_loss'.format(i), loss, grad_idx)


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    data = IRLDataset(config.data)
    writer = SummaryWriter(logdir=config.tensorboard_dir + '/{}_{}_Model'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config.name))
    model = LinearModel(size=2)
    fit(model, data.x, writer, config)
    print('Finished Training')


if __name__ == '__main__':
    config = get_args(config_path='./configs/irl_config.yaml')
    pprint.pprint(config)
    setup_training_output(config)
    main(config)