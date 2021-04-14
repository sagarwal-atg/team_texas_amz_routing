import sys
import os
import datetime

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dist_matrix_dataset import DistMatrixDataset
from models import LinearModel, DoubleLinearModel, ScalingLinearModel
from tsp import compute_tsp_solution
from tensorboardX import SummaryWriter
from IPython import embed


BATCHSIZE = 4
DATASIZE = 1000
NUM_EPOCHS = 1000
MAX_ROUTE_LEN = 250
MAX_COST = 5000.0

print('Loading Data')
base_path = '/Users/Somi/Desktop/Projects/amazon_challenge/'
route_filepath = base_path + 'data/model_build_inputs/route_data.json'
actual_filepath = base_path + 'data/model_build_inputs/actual_sequences.json'
travel_times_filepath = base_path + 'data/model_build_inputs/travel_times.json'

if not os.path.exists(base_path + "trained_models/" + sys.argv[1]):
    os.makedirs(base_path + "trained_models/" + sys.argv[1])


route_dataset = DistMatrixDataset(route_filepath, actual_filepath, travel_times_filepath, MAX_ROUTE_LEN, datasize=DATASIZE, MAX_COST=MAX_COST)
train_dataloader = DataLoader(route_dataset, batch_size=BATCHSIZE, shuffle=True)
test_dataloader = DataLoader(route_dataset, batch_size=1, shuffle=True)
print('Loaded Data')


linear_model = DoubleLinearModel(size=MAX_ROUTE_LEN)
optimizer = optim.SGD(linear_model.parameters(), lr=0.1, momentum=0.9)

writer = SummaryWriter(logdir=base_path + 'runs/{}_{}_Model_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), sys.argv[1], DATASIZE))

print('Starting Training')
loss_idx = 0

linear_model.train()
for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, _, _ = data

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = linear_model(inputs)
        loss = torch.norm(labels - outputs)
        loss.backward()
        optimizer.step()
        
        loss_iter = loss.item()

        # print statistics
        running_loss += loss_iter
        writer.add_scalar('Train/loss', loss_iter, loss_idx)
        loss_idx += 1

    if epoch % 100 == 0:
        torch.save(linear_model.state_dict(), base_path + "trained_models/{}/model_{}.pt".format(sys.argv[1], epoch))
    print("Epoch: {}, Loss: {}".format(epoch, running_loss))

print('Finished Training')




