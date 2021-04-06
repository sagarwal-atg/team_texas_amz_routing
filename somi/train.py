import torch
import datetime
from torch.utils.data import DataLoader, dataset
import torch.optim as optim

from route_dataset import RouteDataset
from models import LinearModel
from tsp import compute_tsp_solution
from tensorboardX import SummaryWriter

BATCHSIZE = 4
DATASIZE = 20
NUM_EPOCHS = 100
MAX_ROUTE_LEN = 200
MAX_COST = 5000.0

print('Loading Data')
base_path = '/Users/Somi/Desktop/Projects/amazon_challenge/'
route_filepath = base_path + 'data/model_build_inputs/route_data.json'
actual_filepath = base_path + 'data/model_build_inputs/actual_sequences.json'
travel_times_filepath = base_path + 'data/model_build_inputs/travel_times.json'

route_dataset = RouteDataset(route_filepath, actual_filepath, travel_times_filepath, MAX_ROUTE_LEN, datasize=DATASIZE, MAX_COST=MAX_COST)
train_dataloader = DataLoader(route_dataset, batch_size=BATCHSIZE, shuffle=True)
test_dataloader = DataLoader(route_dataset, batch_size=1, shuffle=True)
print('Loaded Data')


linear_model = LinearModel(size=MAX_ROUTE_LEN)
optimizer = optim.SGD(linear_model.parameters(), lr=0.001, momentum=0.9)

writer = SummaryWriter(logdir=base_path + 'runs/{}_LinearModel_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), DATASIZE))

print('Starting Training')
loss_idx = 0
for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, _ = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = linear_model(inputs)
        loss = torch.norm(labels - outputs)
        loss.backward()
        optimizer.step()
        
        loss_iter = loss.item() / MAX_COST

        # print statistics
        running_loss += loss_iter
        writer.add_scalar('Train/loss/critic_1', loss_iter, loss_idx)
        loss_idx += 1
    
    print("Epoch: {}, Loss: {}".format(epoch, running_loss))

print('Finished Training')
torch.save(linear_model.state_dict(), base_path + "trained_models/Linear_Model_{}.pt".format(loss_idx))


# for i, data in enumerate(test_dataloader):
#     input, label, stop_keys = data
#     distance_matrix = linear_model(input)
#     distance_matrix = distance_matrix.detach().numpy()
#     distance_matrix = distance_matrix[:len(stop_keys), :len(stop_keys)]
#     tsp_seq = compute_tsp_solution(distance_matrix, stop_keys)

#     num_matches = 0
#     for actual, tsp in zip(stop_keys, tsp_seq):
#         num_matches = int(actual == tsp)
#     print("Route: {}, Num Matches: {}".format(i, num_matches))


