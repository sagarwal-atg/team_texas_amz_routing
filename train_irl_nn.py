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

from dataloaders.irl_dataset import (
    MAX_ROUTE_LEN,
    IRLNNDataset,
    irl_nn_collate,
    seq_binary_mat,
)
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
from models.irl_models import IRLModel, TC_Model
from training_utils.arg_utils import get_args, setup_training_output
from tsp_solvers import constrained_tsp

device = torch.device("cpu")

HIGH_SCORE_GAIN = 1.0
MEDIUM_SCORE_GAIN = 1.0
LOW_SCORE_GAIN = 1.0


class TC_Len_Scheduler:
    def __init__(self, start_len, finish_len, num_epochs):
        self.num_epochs = num_epochs
        self.arr = np.linspace(start_len, finish_len, num_epochs)
        self.curr_len = start_len

    def update(self, curr_epoch_idx):
        if curr_epoch_idx > self.num_epochs:
            self.curr_len = self.arr[-1]
        else:
            self.curr_len = self.arr[curr_epoch_idx]
        print(self.curr_len)


tc_len_scheduler = TC_Len_Scheduler(6.0, 2.0, 20)


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
        violation_up += torch.max(torch.zeros(1), time - time_constraints[seq[idx]][1])
        violation_down += torch.max(
            torch.zeros(1), time_constraints[seq[idx]][0] - time
        )
    return violation_up + violation_down


def tc_tensor_to_tuple_list(tc_tensor):
    tc_list = []
    tc_list_tensor = []
    for route in tc_tensor:
        tc_route_list = []
        tc_route_list_tensor = []
        for tc in route:
            start = torch.max(torch.zeros(1), (tc[0] - tc[1]) * 3600)
            end = (tc[0] + tc[1]) * 3600
            tc_route_list_tensor.append([start, end])
            tc_route_list.append((start.clone().detach(), end.clone().detach()))
        tc_list.append(tc_route_list)
        tc_list_tensor.append(tc_route_list_tensor)

    return tc_list, tc_list_tensor


def compute_tsp_seq_for_route(data, lamb):
    (
        objective_matrix,
        travel_times,
        time_constraints,
        stop_ids,
        travel_time_dict,
        label,
        tc_list,
        depot,
    ) = data

    ###########
    try:
        pred_seq = constrained_tsp.constrained_tsp(
            # objective_matrix * seq_obj_mat * travel_times,
            objective_matrix + travel_times,
            travel_times,
            tc_list,
            depot=depot,
            lamb=int(lamb),
        )
    except AssertionError:
        print("TSP Solution None, Using Raw TSP")
        pred_seq = constrained_tsp.constrained_tsp(
            travel_times,
            travel_times,
            time_constraints,
            depot=depot,
            lamb=int(lamb),
        )

    demo_stop_ids = [stop_ids[j] for j in label]
    demo_stop_ids.append(demo_stop_ids[0])

    pred_stop_ids = [stop_ids[j] for j in pred_seq]
    pred_stop_ids.append(pred_stop_ids[0])

    seq_score = score(demo_stop_ids, pred_stop_ids, travel_time_dict)

    return (pred_seq, seq_score)


def compute_tsp_seq_for_a_batch(batch_data, lamb):
    pool = Pool(processes=multiprocessing.cpu_count())
    tsp_func = partial(compute_tsp_seq_for_route, lamb=lamb)
    batch_output = list(tqdm(pool.imap(tsp_func, batch_data), total=len(batch_data)))
    pool.close()
    return batch_output


def irl_loss(batch_output, thetas_tensor, tc_tensor, tsp_data, model):

    loss = 0.0
    all_pred_tv = 0.0
    all_demo_tv = 0.0
    for route_idx in range(len(batch_output)):
        pred_seq = batch_output[route_idx][0]

        travel_times_tensor = torch.from_numpy(tsp_data[route_idx].travel_times)

        obj_data_tensor = (
            thetas_tensor[route_idx]
            + travel_times_tensor
            # thetas_tensor[route_idx]
            # * seq_tensor[route_idx]
            # * travel_times_tensor
        )

        pred_tv = compute_time_violation_seq(
            travel_times_tensor, tc_tensor[route_idx], pred_seq
        )
        demo_tv = compute_time_violation_seq(
            travel_times_tensor,
            tc_tensor[route_idx],
            tsp_data[route_idx].label,
        )

        pred_cost = torch.sum(
            torch.from_numpy(seq_binary_mat(pred_seq)).type(torch.FloatTensor)
            * obj_data_tensor
        )
        demo_cost = torch.sum(
            torch.from_numpy(tsp_data[route_idx].binary_mat).type(torch.FloatTensor)
            * obj_data_tensor
        )

        lamb = 1.0
        # lamb = model.lamb

        if tsp_data[route_idx].route_score == RouteScoreType.High:
            route_loss = HIGH_SCORE_GAIN * F.relu(
                torch.log(demo_cost + lamb * demo_tv)
                - torch.log(pred_cost + lamb * pred_tv)
            )
        elif tsp_data[route_idx].route_score == RouteScoreType.Low:
            route_loss = LOW_SCORE_GAIN * F.relu(
                torch.log(pred_cost + lamb * pred_tv)
                - torch.log(demo_cost + lamb * demo_tv)
            )

        tc_loss = 0.0
        for jdx, tc in enumerate(tsp_data[route_idx].time_constraints):
            if tc[1] - tc[0] < 3.0 * 3600:
                tc_loss += ((tc[1] - tc[0]) / 2 - tc_tensor[route_idx][jdx][0]) ** 2
                # tc_loss += ((tc[1] - tc[0]) - tc_tensor[route_idx][jdx][1]) ** 2

        loss += route_loss + tc_loss
        all_demo_tv += demo_tv
        all_pred_tv += pred_tv

    loss = loss / len(batch_output)
    all_demo_tv = all_demo_tv / len(all_demo_tv)
    all_pred_tv = all_pred_tv / len(all_pred_tv)

    return (loss, all_pred_tv, all_demo_tv)


def process(models, nn_datas, tsp_data):

    model, seq_model = models
    nn_data, seq_nn_data = nn_datas

    stack_nn_data = torch.cat(nn_data, 0)
    stack_nn_data = stack_nn_data.to(device)
    obj_matrix = model(stack_nn_data)

    stack_seq_data = torch.cat(seq_nn_data, 0)
    stack_seq_data = stack_seq_data.to(device)
    seq_obj_matrix = seq_model(stack_seq_data, tc_len_scheduler.curr_len)

    idx_so_far = 0
    seq_idx_so_far = 0
    thetas_np = []
    thetas_tensor = []
    seq_tensor = []
    # seq_np = []
    for idx, data in enumerate(tsp_data):
        route_len = data.travel_times.shape[0]
        thetas_tensor.append(
            obj_matrix[idx_so_far : (idx_so_far + (route_len * route_len))].reshape(
                (route_len, route_len)
            )
        )

        seq_tensor.append(
            seq_obj_matrix[seq_idx_so_far : (seq_idx_so_far + route_len), :]
        )

        theta = thetas_tensor[idx].clone()
        theta = theta.cpu().detach().numpy()
        thetas_np.append(theta)

        # theta_seq = seq_tensor[idx].clone()
        # theta_seq = theta_seq.cpu().detach().numpy()
        # seq_np.append(theta_seq)

        idx_so_far += route_len * route_len
        seq_idx_so_far += route_len

    tc_list, tc_list_tensor = tc_tensor_to_tuple_list(seq_tensor)

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
                tc_list[idx],
                data.depot,
            )
        )

    # data = compute_tsp_seq_for_route(
    #     batch_data[0], model.get_lambda().clone().detach().numpy()
    # )

    batch_output = compute_tsp_seq_for_a_batch(batch_data, 1.0)

    loss, pred_tv, demo_tv = irl_loss(
        batch_output, thetas_tensor, tc_list_tensor, tsp_data, model
    )

    thetas_norm_sum = 0
    for tht in thetas_tensor:
        thetas_norm_sum += torch.norm(tht)
    thetas_norm = thetas_norm_sum / len(thetas_tensor)

    seq_thetas_norm_sum = 0
    for stht in seq_tensor:
        seq_thetas_norm_sum += torch.norm(stht)
    seq_thetas_norm = seq_thetas_norm_sum / len(seq_tensor)

    return (
        loss,
        batch_output,
        thetas_norm,
        seq_thetas_norm,
        pred_tv,
        demo_tv,
    )


def train(
    models,
    dataloader,
    writer,
    config,
    epoch_idx,
    optimizer,
    scheduler,
    best_loss,
    best_score,
):
    model, seq_model = models
    model.train()
    seq_model.train()

    train_score = []
    train_loss = []

    tc_len_scheduler.update(epoch_idx)

    paths_so_far = 0
    for d_idx, data in enumerate(dataloader):
        start_time = time.time()
        nn_data, tsp_data, scaled_tc_data, seq_nn_data = data
        optimizer.zero_grad()

        (loss, batch_output, thetas_norm, seq_thetas_norm, pred_tv, demo_tv,) = process(
            (model, seq_model),
            (nn_data, seq_nn_data),
            tsp_data,
        )

        paths_so_far += len(batch_output)

        if epoch_idx != 0 or config.train_on_first:
            loss.backward()
            optimizer.step()

        learning_rate = optimizer.param_groups[0]["lr"]

        res = list(zip(*batch_output))
        batch_seq_score = np.array(res[1])

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
            "Epoch: {}, Step: {}, Loss: {:0.2f}, Score: {:0.3f}, Time: {:0.2f} sec,"
            " Lambda: {:0.2f}, LR: {:0.2f}, Theta Norm: {:0.2f}, Seq Thetas Norm: {:0.2f}"
            " Pred tv: {:0.2f}, Demo tv: {:0.2f}".format(
                epoch_idx,
                d_idx,
                loss.item(),
                mean_score,
                time.time() - start_time,
                model.get_lambda().clone().detach().numpy()[0],
                learning_rate,
                thetas_norm,
                seq_thetas_norm,
                pred_tv.item(),
                demo_tv.item(),
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

    return best_loss, best_score


def eval(models, dataloader, writer, config, epoch_idx):

    model, seq_model = models

    model.eval()
    seq_model.eval()

    eval_loss = []
    eval_score = []

    paths_so_far = 0
    for d_idx, data in enumerate(dataloader):
        nn_data, tsp_data, scaled_tc_data, seq_nn_data = data

        (loss, batch_output, thetas_norm, seq_thetas_norm, pred_tv, demo_tv,) = process(
            (model, seq_model),
            (nn_data, seq_nn_data),
            tsp_data,
        )

        res = list(zip(*batch_output))
        batch_seq_score = np.array(res[1])

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

    num_tc_features = 2 + config.data.num_stop_features + config.data.num_route_features
    seq_model = TC_Model(num_features=num_tc_features, out_features=2)
    seq_model = seq_model.to(device)
    print(seq_model)

    if hasattr(config, "model_save_path"):
        chkpt = torch.load(config.model_save_path, map_location=torch.device(device))
        model.load_state_dict(chkpt)
        print(OKBLUE + "Loaded Weights from :{}".format(config.model_save_path) + ENDC)

    if hasattr(config, "seq_save_path"):
        chkpt = torch.load(config.seq_save_path, map_location=torch.device(device))
        seq_model.load_state_dict(chkpt)
        print(OKBLUE + "Loaded Weights from :{}".format(config.seq_save_path) + ENDC)

    params = list(model.parameters()) + list(seq_model.parameters())
    # params = seq_model.parameters()
    optimizer = optim.Adam(params, lr=config.learning_rate)

    def lr_lambda(epch):
        return config.lr_lambda ** epch

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_loss = 1e10
    best_score = 1e10

    for epoch_idx in range(config.num_train_epochs):
        best_loss, best_score = train(
            (model, seq_model),
            train_loader,
            writer,
            config,
            epoch_idx,
            optimizer,
            scheduler,
            best_loss,
            best_score,
        )
        if not epoch_idx % config.eval_iter:
            eval(
                (model, seq_model),
                test_loader,
                writer,
                config,
                epoch_idx,
            )
    print("Finished Training")


if __name__ == "__main__":
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
