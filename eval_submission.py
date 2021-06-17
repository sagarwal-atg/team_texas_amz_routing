#!/usr/bin/env python3
import datetime
import multiprocessing
import pprint
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm

from dataloaders.irl_dataset import (IRL_NN_Eval_Dataset, irl_nn_collate,
                                     seq_binary_mat)
from dataloaders.utils import (ENDC, OKBLUE, OKGREEN, OKRED, OKYELLOW,
                               RouteScoreType, TrainTest)
from eval_utils.score import score
from models.irl_models import IRL_Neighbor_Model, IRLModel
from training_utils.arg_utils import get_args, setup_training_output
from tsp_solvers import constrained_tsp

device = torch.device("cpu")

HIGH_SCORE_GAIN = 1.0
MEDIUM_SCORE_GAIN = 1.0
LOW_SCORE_GAIN = 1.0


def compute_link_cost_seq(time_matrix, link_features, seq):
    time_cost = 0
    feature_cost = np.zeros(link_features.shape[0])
    for idx in range(1, len(seq)):
        time_cost += time_matrix[seq[idx - 1], seq[idx]]
        feature_cost += link_features[:, seq[idx - 1], seq[idx]]
    time_cost += time_matrix[seq[-1]][seq[0]]
    feature_cost += link_features[:, seq[-1], seq[0]]
    return time_cost, feature_cost


def compute_time_violation_seq(time_matrix, time_constraints, seq):
    time = 0.0
    violation_down, violation_up = 0, 0
    for idx in range(1, len(seq)):
        time += time_matrix[seq[idx - 1], seq[idx]]
        violation_up += max(0, time - time_constraints[seq[idx]][1])
        violation_down += max(0, time_constraints[seq[idx]][0] - time)
    return violation_up + violation_down


def compute_tsp_seq_for_route(data, lamb):
    (
        objective_matrix,
        travel_times,
        time_constraints,
        stop_ids,
        travel_time_dict,
    ) = data

    ###########
    try:
        pred_seq = constrained_tsp.constrained_tsp(
            objective_matrix + travel_times,
            travel_times,
            time_constraints,
            depot=label[0],
            lamb=int(lamb),
        )
    except AssertionError:
        print("TSP Solution None, Using Travel Times.")
        pred_seq = constrained_tsp.constrained_tsp(
            travel_times,
            travel_times,
            time_constraints,
            depot=label[0],
            lamb=int(lamb),
        )

    pred_stop_ids = [stop_ids[j] for j in pred_seq]
    pred_stop_ids.append(pred_stop_ids[0])

    return pred_seq


def compute_tsp_seq_for_a_batch(batch_data, lamb):
    pool = Pool(processes=multiprocessing.cpu_count())
    tsp_func = partial(compute_tsp_seq_for_route, lamb=lamb)
    batch_output = list(tqdm(pool.imap(tsp_func, batch_data), total=len(batch_data)))
    pool.close()
    return batch_output


def process(model, nn_data, tsp_data):
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
        theta = theta.cpu().detach().numpy()
        thetas_np.append(theta)

        idx_so_far += route_len * route_len

    batch_data = []
    for idx, data in enumerate(tsp_data):
        batch_data.append(
            (
                thetas_np[idx],
                data.travel_times,
                data.time_constraints,
                data.stop_ids,
                data.travel_time_dict,
            )
        )

    batch_output = compute_tsp_seq_for_a_batch(
        batch_data, model.get_lambda().clone().detach().numpy()
    )

    return batch_output


def eval(model, dataloader, config):
    model.eval()

    eval_loss = []
    eval_score = []

    paths_so_far = 0
    for d_idx, data in enumerate(dataloader):
        nn_data, tsp_data, _ = data

        loss, batch_output, thetas_norm = process(
            model,
            nn_data,
            tsp_data,
        )

        res = list(zip(*batch_output))
        batch_seq_score = np.array(res[3])

        mean_score = np.mean(batch_seq_score)
        eval_loss.append(loss.item())
        eval_score.append(mean_score)

    mean_eval_loss = sum(eval_loss) / len(eval_loss)
    mean_eval_score = sum(eval_score) / len(eval_score)

    print(
        OKGREEN
        + "Eval Loss: {}, Eval Score: {}".format(mean_eval_loss, mean_eval_score)
        + ENDC
    )


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    test_data = IRL_NN_Eval_Dataset(
        config.data, train_or_test=TrainTest.test, cache_path=config.training_dir
    )
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
        print(OKBLUE + "Loaded Weights from :{}".format(config.save_path) + ENDC)

    eval(model, test_loader, config)
    print("Finished Training")


if __name__ == "__main__":
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
