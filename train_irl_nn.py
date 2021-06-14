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

from dataloaders.irl_dataset import IRLNNDataset, irl_nn_collate, seq_binary_mat
from dataloaders.utils import (
    ENDC,
    OKBLUE,
    OKGREEN,
    OKRED,
    OKYELLOW,
    RouteScoreType,
    TrainTest,
)
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
        label,
        prev_pred_path,
    ) = data

    demo_stop_ids = [stop_ids[j] for j in label]
    demo_stop_ids.append(demo_stop_ids[0])

    ###########
    try:
        pred_seq = constrained_tsp.constrained_tsp(
            objective_matrix + travel_times,
            travel_times,
            time_constraints,
            depot=label[0],
            lamb=int(lamb),
            prev_solution=prev_pred_path,
        )
    except AssertionError:
        pred_seq = prev_pred_path
        print("TSP Solution None, Using Prev Path")

    pred_tv = compute_time_violation_seq(travel_times, time_constraints, pred_seq)
    demo_tv = compute_time_violation_seq(travel_times, time_constraints, label)

    pred_stop_ids = [stop_ids[j] for j in pred_seq]
    pred_stop_ids.append(pred_stop_ids[0])

    seq_score = score(demo_stop_ids, pred_stop_ids, travel_time_dict)

    return (pred_seq, demo_tv, pred_tv, seq_score)


def compute_tsp_seq_for_a_batch(batch_data, lamb):
    pool = Pool(processes=multiprocessing.cpu_count())
    tsp_func = partial(compute_tsp_seq_for_route, lamb=lamb)
    batch_output = list(tqdm(pool.imap(tsp_func, batch_data), total=len(batch_data)))
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

        if tsp_data[route_idx].route_score == RouteScoreType.High:
            route_loss = HIGH_SCORE_GAIN * F.relu(
                torch.log(demo_cost + model.lamb * demo_tv)
                - torch.log(pred_cost + model.lamb * pred_tv)
            )
        elif tsp_data[route_idx].route_score == RouteScoreType.Low:
            route_loss = LOW_SCORE_GAIN * F.relu(
                torch.log(pred_cost + model.lamb * pred_tv)
                - torch.log(demo_cost + model.lamb * demo_tv)
            )

        loss += route_loss

    loss = loss / len(batch_output)

    return loss


def process(model, nn_data, tsp_data, train_pred_paths):
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
                data.label,
                train_pred_paths[idx],
            )
        )

    batch_output = compute_tsp_seq_for_a_batch(
        batch_data, model.get_lambda().clone().detach().numpy()
    )

    loss = irl_loss(batch_output, thetas_tensor, tsp_data, model)
    return loss, batch_output


def train(
    model,
    dataloader,
    writer,
    config,
    epoch_idx,
    optimizer,
    scheduler,
    best_loss,
    best_score,
    train_pred_paths,
):

    model.train()

    train_score = []
    train_loss = []

    paths_so_far = 0
    for d_idx, data in enumerate(dataloader):
        start_time = time.time()
        nn_data, tsp_data, scaled_tc_data = data
        optimizer.zero_grad()

        loss, batch_output = process(
            model,
            nn_data,
            tsp_data,
            train_pred_paths[paths_so_far : (paths_so_far + len(tsp_data))],
        )

        for kdx in range(len(batch_output)):
            train_pred_paths[paths_so_far + kdx] = batch_output[kdx][0]

        paths_so_far += len(batch_output)

        if epoch_idx != 0 or config.train_on_first:
            loss.backward()
            optimizer.step()

        learning_rate = optimizer.param_groups[0]["lr"]

        res = list(zip(*batch_output))
        batch_seq_score = np.array(res[3])

        mean_score = np.mean(batch_seq_score)
        train_loss.append(loss.item())
        train_score.append(mean_score)

        writer.add_scalar(
            "Train/batch_route_loss", loss.item(), epoch_idx * len(dataloader) + d_idx
        )
        writer.add_scalar(
            "Train/batch_route_score",
            mean_score,
            epoch_idx * len(dataloader) + d_idx,
        )

        print(
            "Epoch: {}, Step: {}, Loss: {:01f}, Score: {:01f}, Time: {} sec, Lambda: {}, LR: {}".format(
                epoch_idx,
                d_idx,
                loss.item(),
                mean_score,
                time.time() - start_time,
                model.get_lambda().clone().detach().numpy(),
                learning_rate,
            )
        )

    if epoch_idx != 0 or config.train_on_first:
        scheduler.step()

    mean_train_loss = sum(train_loss) / len(train_loss)
    mean_train_score = sum(train_score) / len(train_score)

    loss_print_str = "{}".format(mean_train_loss)
    if best_loss > mean_train_loss:
        best_loss = mean_train_loss
        torch.save(
            model.state_dict(),
            config.training_dir + "/best_model_{}.pt".format(config.name),
        )
        print("Model Saved")
        loss_print_str = OKRED + loss_print_str + ENDC

    score_print_str = "{}".format(mean_train_score)
    if best_score > mean_train_score:
        best_score = mean_train_score
        torch.save(
            model.state_dict(),
            config.training_dir + "/best_score_{}.pt".format(config.name),
        )
        print("Model Saved")
        score_print_str = OKYELLOW + score_print_str + ENDC

    print(
        OKBLUE
        + "Loss: "
        + ENDC
        + loss_print_str
        + OKBLUE
        + " , Score: "
        + ENDC
        + score_print_str
    )

    writer.add_scalar("Train/loss", mean_train_loss, epoch_idx)
    writer.add_scalar("Train/score", mean_train_score, epoch_idx)

    return best_loss, best_score, train_pred_paths


def eval(model, dataloader, writer, config, epoch_idx, test_pred_paths):
    model.eval()

    eval_loss = []
    eval_score = []

    paths_so_far = 0
    for d_idx, data in enumerate(dataloader):
        nn_data, tsp_data, scaled_tc_data = data

        loss, batch_output = process(
            model,
            nn_data,
            tsp_data,
            test_pred_paths[paths_so_far : (paths_so_far + len(tsp_data))],
        )

        for kdx in range(len(batch_output)):
            test_pred_paths[paths_so_far + kdx] = batch_output[kdx][0]

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
    writer.add_scalar("Eval/loss", mean_eval_loss, epoch_idx)
    writer.add_scalar("Eval/score", mean_eval_score, epoch_idx)

    return test_pred_paths


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_data = IRLNNDataset(
        config.data, train_or_test=TrainTest.train, cache_path=config.training_dir
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        collate_fn=irl_nn_collate,
        num_workers=2,
    )

    test_data = IRLNNDataset(
        config.data, train_or_test=TrainTest.test, cache_path=config.training_dir
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
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
        print(OKBLUE + "Loaded Weights from :{}".format(config.save_path) + ENDC)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    def lr_lambda(epch):
        return config.lr_lambda ** epch

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_loss = 1e10
    best_score = 1e10

    train_pred_paths = [None] * int(config.data.train_split * config.data.slice_end)
    test_pred_paths = [None] * int(
        (1 - config.data.train_split) * config.data.slice_end
    )

    for epoch_idx in range(config.num_train_epochs):
        best_loss, best_score, train_pred_paths = train(
            model,
            train_loader,
            writer,
            config,
            epoch_idx,
            optimizer,
            scheduler,
            best_loss,
            best_score,
            train_pred_paths,
        )
        if not epoch_idx % config.eval_iter:
            test_pred_paths = eval(
                model, test_loader, writer, config, epoch_idx, test_pred_paths
            )
    print("Finished Training")


if __name__ == "__main__":
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
