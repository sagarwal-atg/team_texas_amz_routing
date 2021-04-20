import os
import torch
from models.models import ARC_Classifier
from dataloaders.irl_dataset import IRLDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
from easydict import EasyDict as edict
import yaml
import argparse

null_callback = lambda *args, **kwargs: None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fit(model, dataloader, epochs=1, verbose=0,
        cb_after_batch_update=null_callback, cb_after_epoch=null_callback):

    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_loss = []
        for data in dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            loss = model.train_on_batch(inputs, labels)
            epoch_loss.append(loss.cpu().detach())
            cb_after_batch_update(loss)
        cb_after_epoch(epoch, model)
        if verbose > 0:
            accuracy = (model(inputs).argmax(1) == labels).float().mean().item()
            print(f'Epoch: {epoch}, Loss {np.mean(epoch_loss):.4f}, Accuracy: {accuracy:.2f}')


def main(paths, batch_size, epochs, learning_rate):
    data = IRLDataset(paths, slice_end=800)
    train_size = int(len(data)*.7)
    test_size = len(data) - train_size
    train, test = random_split(data, [train_size, test_size])
    print(f'Train size: {len(train)}, Test size: {len(test)}')
    train_loader = DataLoader(train, batch_size, shuffle=True)

    def test_cb(epoch, model):
        inputs, labels = test[:]
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        accuracy = (model(inputs).argmax(1) == labels).float().mean().item()
        loss = model.get_loss(outputs, labels).cpu()
        print(f'Epoch: {epoch}, Test Loss {loss:.4f}, Test Accuracy: {accuracy:.2f}')

    model = ARC_Classifier(
        data.max_route_len,
        data.num_features,
        hidden_sizes=[256],
        lr=learning_rate,
    ).to(device)

    fit(model, train_loader, epochs, verbose=1, cb_after_epoch=test_cb)
    print('Finished Training')

def test(paths):
    data = IRLDataset(paths, slice_end=800)
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


def get_args(config_path='./configs/config.yaml'):
    global device
    parser = argparse.ArgumentParser(description='Training code')
    config = edict(yaml.safe_load(open(config_path, 'r')))

    parser.add_argument('--batchsize', default=config.batchsize, type=int)
    parser.add_argument('--epochs', default=config.num_train_epochs, type=int)
    parser.add_argument('--datapath', default=config.base_path, type=str, help='base path to the data')
    parser.add_argument('--lr', default=config.learning_rate, type=str)
    parser.add_argument('--device', default=device, type=str, help='cpu or gpu')

    args = parser.parse_args()
    device = args.device

    paths = edict(
        route = os.path.join(args.datapath, config.route_filename),
        sequence = os.path.join(args.datapath, config.sequence_filename),
        travel_time = os.path.join(args.datapath, config.travel_times_filename),
        packages = os.path.join(args.datapath, config.package_data_filename),
    )

    return paths, args.batchsize, args.epochs, args.lr


if __name__ == '__main__':
    paths, batch_size, epochs, lr = get_args()
    main(paths, batch_size, epochs, lr)