import sys
import torch
import datetime
from torch.utils.data import DataLoader

from route_dataset import RouteDataset
from models import LinearModel, DoubleLinearModel, ScalingLinearModel
from tsp import compute_tsp_solution
import editdistance

from IPython import embed

BATCHSIZE = 4
DATASIZE = 10
MAX_ROUTE_LEN = 250
MAX_COST = 5000.0

print('Loading Data')
base_path = '/Users/Somi/Desktop/Projects/amazon_challenge/'
route_filepath = base_path + 'data/model_build_inputs/route_data.json'
actual_filepath = base_path + 'data/model_build_inputs/actual_sequences.json'
travel_times_filepath = base_path + 'data/model_build_inputs/travel_times.json'

route_dataset = RouteDataset(route_filepath, actual_filepath, travel_times_filepath, MAX_ROUTE_LEN, datasize=DATASIZE, MAX_COST=MAX_COST)
test_dataloader = DataLoader(route_dataset, batch_size=1, shuffle=False)
print('Loaded Data')

model_path = base_path + "trained_models/" + sys.argv[1]
linear_model = ScalingLinearModel(size=MAX_ROUTE_LEN)
linear_model.load_state_dict(torch.load(model_path))
linear_model.eval()


def compute_tsp_and_edit_distance(distance_matrix, actual_stop_seq):
    distance_matrix = distance_matrix.detach().numpy()
    distance_matrix = distance_matrix[0, :len(actual_stop_seq), :len(actual_stop_seq)]
    tsp_seq = compute_tsp_solution(distance_matrix, actual_stop_seq)

    sorted_actual_stop_seq = [k for k, v in sorted(actual_stop_seq.items(), key=lambda item: item[1])]
    return editdistance.eval(tsp_seq, sorted_actual_stop_seq)


print('Evaluating Model')
orig_edit_distances = []
model_edit_distance = []
for i, data in enumerate(test_dataloader):
    input, _, stop_key, distance_matrix = data
    actual_stop_seq = test_dataloader.dataset.stop_dict[stop_key[0]]
    orig_edit_distances.append(compute_tsp_and_edit_distance(distance_matrix, actual_stop_seq))

    distance_matrix = linear_model(input)
    model_edit_distance.append(compute_tsp_and_edit_distance(distance_matrix, actual_stop_seq))

avg_distance = 0
for i in range(DATASIZE):
    avg_distance += abs(orig_edit_distances[i] - model_edit_distance[i])
avg_distance = avg_distance / DATASIZE

print("Average Difference: {}".format(avg_distance))
print("Average Edit Distance Original: {}".format(sum(orig_edit_distances) / len(orig_edit_distances)))
print("Average Edit Distance Linear Model: {}".format(sum(model_edit_distance) / len(model_edit_distance)))






