import argparse
import datetime
import os
import pprint

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from IPython.terminal.embed import embed
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split

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
from models.models import ARC_Classifier
from training_utils.arg_utils import get_args, setup_training_output

null_callback = lambda *args, **kwargs: None


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    print(torch.unique(y_pred_tag))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def fit(
    model,
    dataloader,
    writer,
    optimizer,
    config,
    verbose=0,
    cb_after_batch_update=null_callback,
    cb_after_epoch=null_callback,
):
    model.train()
    train_loss_idx = 0
    test_loss_idx = 0
    best_loss = np.inf

    for epoch in range(config.num_train_epochs):  # loop over the dataset multiple times
        epoch_loss = []
        epoch_acc = []
        for idx, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            _, _, _, inputs, labels = data

            stack_seq_data = torch.cat(inputs, 0)
            stack_label_data = torch.cat(labels, 0)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(stack_seq_data)
            print(outputs)
            # loss = model.get_loss(outputs, stack_label_data)

            loss = model.criterion(outputs, stack_label_data)
            acc = binary_acc(outputs, stack_label_data)

            loss.backward()
            optimizer.step()

            # epoch_loss.append(loss.detach())
            cb_after_batch_update(loss)

            writer.add_scalar("Train/loss", loss.item(), train_loss_idx)
            train_loss_idx += 1

            print(
                f"Epoch: {epoch}, Step: {idx}, Loss {loss.item():.4f}, Accuracy: {acc.item():.2f}"
            )

            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())

        cb_after_epoch(epoch, model, test_loss_idx)
        test_loss_idx += 1

        mean_loss = np.mean(epoch_loss)
        mean_accuracy = np.mean(epoch_acc)

        if best_loss > mean_loss:
            best_loss = mean_loss
            torch.save(
                model.state_dict(),
                config.training_dir + "/model_{}.pt".format(config.name),
            )

        # accuracy = (
        #     (model(stack_seq_data).argmax(1) == stack_label_data).float().mean().item()
        # )
        writer.add_scalar("Train/Accuracy", mean_accuracy, train_loss_idx)

        if verbose > 0:
            print(
                f"Epoch: {epoch}, Loss {mean_loss:.4f}, Accuracy: {mean_accuracy:.2f}"
            )


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # data = ClassificationDataset(config.data)
    # train_size = int(len(data) * config.train_split)
    # test_size = len(data) - train_size
    # train, test = random_split(data, [train_size, test_size])
    # print(f"Train size: {len(train)}, Test size: {len(test)}")
    # train_loader = DataLoader(train, config.batch_size, shuffle=True)

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

    model = ARC_Classifier(
        2 + config.data.num_stop_features + config.data.num_route_features,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def test_cb(epoch, model, loss_idx):
        model.eval()

        # inputs, labels = test[:]
        with torch.no_grad():
            for d_idx, data in enumerate(test_loader):
                _, _, _, inputs, labels = data
                # nn_data, tsp_data, scaled_tc_data, seq_nn_data = data

                stack_seq_data = torch.cat(inputs, 0)
                stack_label_data = torch.cat(labels, 0)

                outputs = model(stack_seq_data)

                # accuracy = (
                #     (model(stack_seq_data).argmax(1) == stack_label_data)
                #     .float()
                #     .mean()
                #     .item()
                # )
                # loss = model.get_loss(outputs, stack_label_data)
                loss = model.criterion(outputs, stack_label_data)
                acc = binary_acc(outputs, stack_label_data)

                print(f"Epoch: {epoch}, Test Loss {loss:.4f}, Test Accuracy: {acc:.2f}")

    fit(
        model,
        train_loader,
        writer,
        optimizer,
        config,
        verbose=1,
        cb_after_epoch=test_cb,
    )
    print("Finished Training")


def test(config):
    data = IRLDataset(config)
    eq = lambda a, b: torch.all(a.eq(b))
    # y should be [2, 0, 1] meaning we go [A->C, B->A, C->B]
    assert eq(data.y, torch.LongTensor([2, 0, 1]))
    # the first row of x should be [0,0,2/3,1,1/3,0] representing this:
    # [time(A->A), zone_cross(A->A), time(A->B), zone_cross(A->B), time(A->C), zone_cross(A->C)]
    # zone_cross is binary and time is being divided by 3 because we normalize time by row sum
    assert eq(
        data.x,
        torch.FloatTensor(
            [
                [0, 0, 2 / 3, 1, 1 / 3, 0],
                [1 / 3, 1, 0, 0, 2 / 3, 1],
                [2 / 3, 0, 1 / 3, 1, 0, 0],
            ]
        ),
    )


if __name__ == "__main__":
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)
