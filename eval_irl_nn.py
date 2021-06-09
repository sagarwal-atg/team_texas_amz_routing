#!/usr/bin/env python3
import datetime
import pprint
import time
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed
from numpy.core.defchararray import mod
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm

from dataloaders.irl_dataset import IRLNNDataset, irl_nn_collate
from dataloaders.utils import ENDC, OKBLUE
from models.irl_models import IRLModel
from train_irl_nn import compute_tsp_seq_for_a_batch, irl_loss
from training_utils.arg_utils import get_args, setup_training_output

device = torch.device("cpu")


def eval(model, dataloader, config):

    model.eval()

    # loop over the dataset multiple times
    epoch_score = 0
    epoch_loss = 0

    for d_idx, data in enumerate(tqdm(dataloader)):
        start_time = time.time()
        nn_data, tsp_data, scaled_tc_data = data

        stack_nn_data = torch.cat(nn_data, 0)
        stack_nn_data = stack_nn_data.to(device)
        obj_matrix = model(stack_nn_data)

        idx_so_far = 0
        thetas_np = []
        thetas_tensor = []
        for idx, data in enumerate(tsp_data):
            route_len = data.travel_times.shape[0]
            thetas_tensor.append(
                obj_matrix[idx_so_far : (idx_so_far + (route_len * route_len))].reshape(
                    (route_len, route_len)
                )
            )

            theta = thetas_tensor[idx].clone()
            theta = theta.detach().numpy()
            thetas_np.append(theta)

            idx_so_far += route_len * route_len

        for theta in thetas_tensor:
            theta.retain_grad()

        batch_data = []
        for idx, data in enumerate(tsp_data):
            batch_data.append(
                (
                    thetas_np[idx],
                    data.travel_times,
                    data.time_constraints,
                    data.stop_ids,
                    data.travel_time_dict,
                    data.label,
                )
            )

        batch_output = compute_tsp_seq_for_a_batch(
            batch_data, model.get_lambda().clone().detach().numpy()
        )

        loss = irl_loss(batch_output, thetas_tensor, tsp_data, model)

        res = list(zip(*batch_output))
        batch_seq_score = np.array(res[3])

        mean_score = np.mean(batch_seq_score)
        epoch_loss += loss.item()
        epoch_score += mean_score

        print(
            "Step: {}, Loss: {:01f}, Score: {:01f}, Time: {} sec, Lambda: {}".format(
                d_idx,
                loss.item(),
                mean_score,
                time.time() - start_time,
                model.get_lambda().clone().detach().numpy(),
            )
        )

    mean_epoch_loss = (epoch_loss * config.batch_size) / (len(dataloader.dataset))
    mean_epoch_score = (epoch_score * config.batch_size) / (len(dataloader.dataset))

    print(
        OKBLUE
        + "Mean Loss: {}, Mean Score: {}".format(mean_epoch_loss, mean_epoch_score)
        + ENDC
    )


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    test_data = IRLNNDataset(config.data, cache_path=config.training_dir)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config.batch_size,
        collate_fn=irl_nn_collate,
        num_workers=2,
    )

    num_features = (
        1 + config.data.num_link_features + config.data.num_route_features
    ) * (config.data.num_neighbors + 1)

    model = IRLModel(num_features=num_features)
    model = model.to(device)
    print(model)

    if hasattr(config, "save_path"):
        chkpt = torch.load(config.save_path, map_location=torch.device(device))
        model.load_state_dict(chkpt)

    eval(model, test_loader, config)
    print("Finished Training")


if __name__ == "__main__":
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
