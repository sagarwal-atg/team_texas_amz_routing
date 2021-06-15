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
from dataloaders.utils import ENDC, OKBLUE, OKGREEN, TrainTest
from models.irl_models import IRLModel
from train_irl_nn import process
from training_utils.arg_utils import get_args, setup_training_output

device = torch.device("cpu")


def eval(model, dataloader, config, epoch_idx, test_pred_paths):
    model.eval()

    eval_loss = []
    eval_score = []

    paths_so_far = 0
    for d_idx, data in enumerate(dataloader):
        nn_data, tsp_data, scaled_tc_data = data

        loss, batch_output, thetas_norm = process(
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

    return test_pred_paths


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    test_data = IRLNNDataset(
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

    test_pred_paths = [None] * int(
        (1 - config.data.train_split) * config.data.slice_end
    )

    test_pred_paths = eval(
        model,
        test_loader,
        config,
        0,
        test_pred_paths,
    )
    print("Finished Training")


if __name__ == "__main__":
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
