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
model = DoubleLinearModel(size=MAX_ROUTE_LEN)
model.load_state_dict(torch.load(model_path))
model.eval()


def longest_common_substring(s1, s2):
   m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
   longest, x_longest = 0, 0
   for x in range(1, 1 + len(s1)):
       for y in range(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
   return s1[x_longest - longest: x_longest]


def test_best_lcs(seq, seq_to_reverse):
    lcs = longest_common_substring(seq, seq_to_reverse)
    reverse_lcs = longest_common_substring(seq, list(reversed(seq_to_reverse)))
    if len(lcs) < len(reverse_lcs):
        lcs = reverse_lcs
    return lcs


def compute_tsp_and_edit_distance(distance_matrix, gt_stop_seq):
    distance_matrix = distance_matrix.detach().numpy()
    distance_matrix = distance_matrix[0, :len(gt_stop_seq), :len(gt_stop_seq)]
    tsp_seq = compute_tsp_solution(distance_matrix, gt_stop_seq)

    assert tsp_seq is not None

    sorted_gt_stop_seq = [k for k, v in sorted(gt_stop_seq.items(), key=lambda item: item[1])]
    edit_dist = min(editdistance.eval(tsp_seq, sorted_gt_stop_seq),
                    editdistance.eval(tsp_seq, list(reversed(sorted_gt_stop_seq))))
    return tsp_seq, sorted_gt_stop_seq, edit_dist


print('Evaluating Model')
orig_edit_distances = []
model_edit_distances = []
label_edit_distances = []

orig_lcs = []
model_lcs = []
label_lcs = []

for i, data in enumerate(test_dataloader):
    input, output, stop_key, distance_matrix = data
    gt_stop_seq = test_dataloader.dataset.stop_dict[stop_key[0]]

    tsp_seq, sorted_gt_stop_seq, gt_edit_dist = compute_tsp_and_edit_distance(distance_matrix, gt_stop_seq)
    orig_edit_distances.append(gt_edit_dist)
    orig_lcs.append(test_best_lcs(sorted_gt_stop_seq, tsp_seq))

    distance_matrix = model(input)
    model_tsp_seq, _, model_edit_dist = compute_tsp_and_edit_distance(distance_matrix, gt_stop_seq)
    model_edit_distances.append(model_edit_dist)
    model_lcs.append(test_best_lcs(sorted_gt_stop_seq, model_tsp_seq))

    label_tsp_seq, _, label_edit_dist = compute_tsp_and_edit_distance(output, gt_stop_seq)
    label_edit_distances.append(label_edit_dist)
    label_lcs.append(test_best_lcs(sorted_gt_stop_seq, model_tsp_seq))

    embed()
    break

# avg_distance = 0
# for i in range(DATASIZE):
#     avg_distance += abs(orig_edit_distances[i] - model_edit_distances[i])
# avg_distance = avg_distance / DATASIZE

# print("Average Difference: {}".format(avg_distance))
# print("Average Edit Distance Original: {}".format(sum(orig_edit_distances) / len(orig_edit_distances)))
# print("Average Edit Distance Linear Model: {}".format(sum(model_edit_distances) / len(model_edit_distances)))






