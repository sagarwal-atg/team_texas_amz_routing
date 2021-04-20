import os
import datetime
from easydict import EasyDict as edict
import argparse
import yaml
import pprint

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataloaders.irl_dataset import IRLDataset
from models.models import IRLLinearModel
from tensorboardX import SummaryWriter
from IPython import embed


parser = argparse.ArgumentParser(description='Training code')
parser.add_argument('--config', default='config.yaml', type=str, help='yaml config file')
args = parser.parse_args()
config = edict(yaml.safe_load(open(args.config, 'r')))
print('==> CONFIG is: \n')
pprint.pprint(config)

print('Loading Data')

if not os.path.exists(config.base_path + "trained_models/" + config.name):
    os.makedirs(config.base_path + "trained_models/" + config.name)

irl_dataset = IRLDataset(config)
train_dataloader = DataLoader(irl_dataset, batch_size=config.batchsize, shuffle=True)
print('Loaded Data')

link_features = irl_dataset.link_features
route_features = irl_dataset.route_features
model = IRLLinearModel(max_route_len=config.max_route_len, link_features_size=len(link_features), route_features_size=len(route_features))
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

writer = SummaryWriter(logdir=config.base_path + 'runs/{}_{}_Model_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config.name, config.datasize))

print('Starting Training')
loss_idx = 0

model.train()
for epoch in range(config.num_train_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        x, actual_route_matrix, _ = data

        with torch.autograd.set_detect_anomaly(True):
            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(x)
            loss = torch.norm(actual_route_matrix - output)
            loss.backward()
            optimizer.step()
        
        loss_iter = loss.item()

        # print statistics
        running_loss += loss_iter
        writer.add_scalar('Train/loss', loss_iter, loss_idx)
        loss_idx += 1

    if epoch % 100 == 0:
        print("Model Saved")
        torch.save(model.state_dict(), config.base_path + "trained_models/{}/model_{}.pt".format(config.name, epoch))
    print("Epoch: {}, Loss: {}".format(epoch, running_loss))

print('Finished Training')




