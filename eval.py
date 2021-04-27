import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from IPython import embed
import pprint

from dataloaders.irl_dataset import ClassificationDataset
from dataloaders.dist_matrix_dataset import DistMatrixDataset
from models.models import ARC_Classifier
from eval_utils.score import score
from tsp_solvers.tsp import compute_tsp_solution
from training_utils.arg_utils import get_args


def compute_tsp(distance_matrix, gt_stop_seq):
    stop_order = list(gt_stop_seq.values())
    tsp_seq = compute_tsp_solution(distance_matrix, gt_stop_seq, depot=stop_order.index(0))

    assert tsp_seq is not None

    sorted_gt_stop_seq = [k for k, v in sorted(gt_stop_seq.items(), key=lambda item: item[1])]
    return tsp_seq, sorted_gt_stop_seq


def main(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data = ClassificationDataset(config.data)
    test_loader = DataLoader(data, 1, shuffle=False)
    route_ids = data.route_ids
    route_lengths = data.route_lengths
    orig_data = DistMatrixDataset(config.data, route_ids)
    stop_seqs, tt_matrices, tt_dicts = orig_data.get_stops_tt_matrices()

    model = ARC_Classifier(
        data.max_route_len,
        data.num_features,
        hidden_sizes=[config.model.hidden_size],
    )
    model_path = os.path.join(config.training_dir, 'model_{}.pt'.format(config.name))
    model.load_state_dict(torch.load(model_path))

    it = iter(test_loader)
    seq_scores = []
    model_scores = []
    for route_num in range(100):
        route_len = int(route_lengths[int(route_num)])
        tt_matrix = np.ones((route_len, route_len))
        data_idx = 0
        while data_idx < route_len:
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = next(it)
            output = model(inputs)
            tt_matrix[data_idx, :] = tt_matrix[data_idx, :] - output.detach().numpy()[0, :route_len]

            data_idx += 1
            
        model_seq, sorted_gt_seq = compute_tsp(tt_matrix, stop_seqs[int(route_num)])
        tsp_seq, sorted_gt_seq = compute_tsp(tt_matrices[int(route_num)], stop_seqs[int(route_num)])
        tsp_seq.append(tsp_seq[0])
        sorted_gt_seq.append(sorted_gt_seq[0])
        model_seq.append(model_seq[0])
        embed()
        seq_score = score(sorted_gt_seq, tsp_seq, tt_dicts[int(route_num)])
        model_score = score(sorted_gt_seq, model_seq, tt_dicts[int(route_num)])
        seq_scores.append(seq_score)
        model_scores.append(model_score)
    
    embed()


if __name__ == '__main__':
    config = get_args()
    pprint.pprint(config)
    main(config)