#!/usr/bin/env python3
import pprint

import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed

from dataloaders.irl_dataset import IRLNNDataset, irl_nn_collate
from dataloaders.utils import ENDC, OKBLUE, OKGREEN, TrainTest
from models.irl_models import IRLModel
from train_irl_nn import process
from training_utils.arg_utils import get_args, setup_training_output

device = torch.device("cpu")


def eval(model, dataloader, config, epoch_idx):
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
            use_replay=False
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
    pprint.pprint(model)

    if hasattr(config, "save_path"):
        chkpt = torch.load(config.save_path, map_location=torch.device(device))
        model.load_state_dict(chkpt)
        print(OKBLUE + "Loaded Weights from :{}".format(config.save_path) + ENDC)

    eval(
        model,
        test_loader,
        config,
        0,
    )
    print("Finished Training")


if __name__ == "__main__":
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
