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

from models.models import ARC_Classifier
from dataloaders.irl_dataset import IRLDataset
from training_utils.arg_utils import get_args, setup_training_output


null_callback = lambda *args, **kwargs: None


def fit(model, dataloader, writer, optimizer, config, verbose=0,
        cb_after_batch_update=null_callback, cb_after_epoch=null_callback):

    train_loss_idx = 0
    test_loss_idx = 0
    best_loss = np.inf

    for epoch in range(config.num_train_epochs):  # loop over the dataset multiple times
        epoch_loss = []
        for data in dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.get_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.detach())
            cb_after_batch_update(loss)
            writer.add_scalar('Train/loss', loss.item(), train_loss_idx)
            train_loss_idx += 1

        cb_after_epoch(epoch, model, test_loss_idx)
        test_loss_idx += 1

        mean_loss = np.mean(epoch_loss)
        if best_loss > mean_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), config.training_dir + "model_{}.pt".format(config.name, epoch))

        if verbose > 0:
            accuracy = (model(inputs).argmax(1) == labels).float().mean().item()
            writer.add_scalar('Train/Accuracy', accuracy, train_loss_idx)
            print(f'Epoch: {epoch}, Loss {mean_loss:.4f}, Accuracy: {accuracy:.2f}')


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data = IRLDataset(config.data)
    train_size = int(len(data)*config.train_split)
    test_size = len(data) - train_size
    train, test = random_split(data, [train_size, test_size])
    print(f'Train size: {len(train)}, Test size: {len(test)}')
    train_loader = DataLoader(train, config.batch_size, shuffle=True)
    
    writer = SummaryWriter(logdir=config.tensorboard_dir + '{}_{}_Model'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config.name))

    model = ARC_Classifier(
        data.max_route_len,
        data.num_features,
        hidden_sizes=[config.model.hidden_size],
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def test_cb(epoch, model, loss_idx):
        inputs, labels = test[:]
        with torch.no_grad():
            outputs = model(inputs)
        accuracy = (model(inputs).argmax(1) == labels).float().mean().item()
        loss = model.get_loss(outputs, labels)
        writer.add_scalar('Test/loss', loss, loss_idx)
        writer.add_scalar('Test/accuracy', accuracy, loss_idx)
        print(f'Epoch: {epoch}, Test Loss {loss:.4f}, Test Accuracy: {accuracy:.2f}')

    fit(model, train_loader, writer, optimizer, config, verbose=1, cb_after_epoch=test_cb)
    print('Finished Training')

def test(config):
    data = IRLDataset(config)
    eq = lambda a, b: torch.all(a.eq(b))
    # y should be [2, 0, 1] meaning we go [A->C, B->A, C->B]
    assert eq(data.y, torch.LongTensor([2, 0, 1]))
    # the first row of x should be [0,0,2/3,1,1/3,0] representing this:
    # [time(A->A), zone_cross(A->A), time(A->B), zone_cross(A->B), time(A->C), zone_cross(A->C)]
    # zone_cross is binary and time is being divided by 3 because we normalize time by row sum
    assert eq(data.x, torch.FloatTensor([
        [0,0,2/3,1,1/3,0],
        [1/3,1,0,0,2/3,1],
        [2/3,0,1/3,1,0,0]]))


if __name__ == '__main__':
    config = get_args()
    pprint.pprint(config)
    setup_training_output(config)
    main(config)