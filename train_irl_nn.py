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

from dataloaders.irl_dataset import (IRLNNDataset, irl_nn_collate,
                                     seq_binary_mat)
from dataloaders.utils import ENDC, OKBLUE, OKRED, OKYELLOW
from eval_utils.score import score
from models.irl_models import IRLModel
from training_utils.arg_utils import get_args, setup_training_output
from tsp_solvers import constrained_tsp

device = torch.device("cpu")


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
        violation_up += max(0, time - time_constraints[idx][1])
        violation_down += max(0, time_constraints[idx][0] - time)
    return violation_up + violation_down


def compute_tsp_seq_for_route(data, lamb):
    (
        objective_matrix,
        travel_times,
        time_constraints,
        stop_ids,
        travel_time_dict,
        label,
    ) = data

    demo_stop_ids = [stop_ids[j] for j in label]
    demo_stop_ids.append(demo_stop_ids[0])

    ###########
    pred_seq = constrained_tsp.constrained_tsp(
        objective_matrix + travel_times,
        travel_times,
        time_constraints,
        depot=label[0],
        lamb=int(lamb),
    )

    pred_tv = compute_time_violation_seq(travel_times, time_constraints, pred_seq)
    demo_tv = compute_time_violation_seq(travel_times, time_constraints, label)

    pred_stop_ids = [stop_ids[j] for j in pred_seq]
    pred_stop_ids.append(pred_stop_ids[0])

    seq_score = score(demo_stop_ids, pred_stop_ids, travel_time_dict)

    return (pred_seq, demo_tv, pred_tv, seq_score)


def compute_tsp_seq_for_a_batch(batch_data, lamb):
    pool = Pool(processes=12)
    tsp_func = partial(compute_tsp_seq_for_route, lamb=lamb)
    batch_output = []
    for out in tqdm(pool.imap(tsp_func, batch_data)):
        batch_output.append(out)
    pool.close()
    return batch_output


def irl_loss(batch_output, thetas_tensor, tsp_data, model):

    loss = 0.0
    for route_idx in range(len(batch_output)):
        pred_seq = batch_output[route_idx][0]
        demo_tv = batch_output[route_idx][1]
        pred_tv = batch_output[route_idx][2]

        travel_times_tensor = torch.from_numpy(tsp_data[route_idx].travel_times)

        pred_cost = torch.sum(
            torch.from_numpy(seq_binary_mat(pred_seq)).type(torch.FloatTensor)
            * (thetas_tensor[route_idx] + travel_times_tensor)
        )
        demo_cost = torch.sum(
            torch.from_numpy(tsp_data[route_idx].binary_mat).type(torch.FloatTensor)
            * (thetas_tensor[route_idx] + travel_times_tensor)
        )

        # route_loss = F.relu(
        #     (demo_cost + model.get_lambda() * demo_tv) - (pred_cost + model.get_lambda() * pred_tv)
        # )

        route_loss = F.relu(
            torch.log(demo_cost + model.lamb * demo_tv)
            - torch.log(pred_cost + model.lamb * pred_tv)
        )

        loss += route_loss

    loss = loss / len(batch_output)

    return loss


def fit(model, dataloader, writer, config):

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()

    best_loss = 1e10
    best_score = 1e10

    # loop over the dataset multiple times
    for epoch_idx in range(config.num_train_epochs):
        epoch_score = 0
        epoch_loss = 0

        for d_idx, data in enumerate(dataloader):
            start_time = time.time()
            nn_data, tsp_data, scaled_tc_data = data
            optimizer.zero_grad()

            stack_nn_data = torch.cat(nn_data, 0)
            stack_nn_data = stack_nn_data.to(device)
            obj_matrix = model(stack_nn_data)

            idx_so_far = 0
            thetas_np = []
            thetas_tensor = []
            for idx, data in enumerate(tsp_data):
                route_len = data.travel_times.shape[0]
                thetas_tensor.append(
                    obj_matrix[
                        idx_so_far : (idx_so_far + (route_len * route_len))
                    ].reshape((route_len, route_len))
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

            if epoch_idx != 0:
                loss.backward()
                optimizer.step()

            res = list(zip(*batch_output))
            batch_seq_score = np.array(res[3])

            mean_score = np.mean(batch_seq_score)
            epoch_loss += loss.item()
            epoch_score += mean_score

            writer.add_scalar(
                "Train/batch_route_loss", loss.item(), epoch_idx * len(dataloader) + idx
            )
            writer.add_scalar(
                "Train/batch_route_score",
                mean_score,
                epoch_idx * len(dataloader) + idx,
            )

            print(
                "Epoch: {}, Step: {}, Loss: {:01f}, Score: {:01f}, Time: {} sec, Lambda: {}".format(
                    epoch_idx,
                    d_idx,
                    loss.item(),
                    mean_score,
                    time.time() - start_time,
                    model.get_lambda().clone().detach().numpy(),
                )
            )

        mean_epoch_loss = (epoch_loss * config.batch_size) / (len(dataloader.dataset))
        mean_epoch_score = (epoch_score * config.batch_size) / (len(dataloader.dataset))

        loss_print_str = "{}".format(mean_epoch_loss)
        if best_loss > mean_epoch_loss:
            best_loss = mean_epoch_loss
            torch.save(
                model.state_dict(),
                config.training_dir + "/best_model_{}.pt".format(config.name),
            )
            print("Model Saved")
            loss_print_str = OKRED + loss_print_str + ENDC

        score_print_str = "{}".format(mean_epoch_score)
        if best_score > mean_epoch_score:
            best_score = mean_epoch_score
            torch.save(
                model.state_dict(),
                config.training_dir + "/best_score_{}.pt".format(config.name),
            )
            print("Model Saved")
            score_print_str = OKYELLOW + score_print_str + ENDC

        print(
            OKBLUE
            + "Epoch Loss: "
            + ENDC
            + loss_print_str
            + OKBLUE
            + " , Epoch Score: "
            + ENDC
            + score_print_str
        )

        writer.add_scalar("Train/loss", mean_epoch_loss, epoch_idx)
        writer.add_scalar("Train/score", mean_epoch_score, epoch_idx)


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_data = IRLNNDataset(config.data, cache_path=config.training_dir)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        collate_fn=irl_nn_collate,
        num_workers=2,
    )

    writer = SummaryWriter(
        logdir=config.tensorboard_dir
        + "/{}_{}_Model".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config.name
        )
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

    fit(model, train_loader, writer, config)
    print("Finished Training")


if __name__ == "__main__":
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
