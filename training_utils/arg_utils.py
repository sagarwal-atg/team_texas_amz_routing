import argparse
from easydict import EasyDict as edict
import yaml
import os


def edict2dict(edict_obj):
    dict_obj = {}
    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals
    return dict_obj


def get_args():
    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument('--config_path', default='./configs/config.yaml', type=str, help='yaml config file')
    parser.add_argument('--data_path', default=None, type=str, help='base path to the data')
    args = parser.parse_args()

    config = edict(yaml.safe_load(open(args.config_path, 'r')))
    
    if args.data_path:
        config.data.base_path = args.data_path
    
    def data_file_path(filename):
        return os.path.join(config.base_path, config.data.base_path, filename)

    config.data.route_path = data_file_path(config.data.route_filename)
    config.data.sequence_path = data_file_path(config.data.sequence_filename)
    config.data.travel_time_path = data_file_path(config.data.travel_times_filename)
    config.data.package_path = data_file_path(config.data.package_data_filename)

    return config


def setup_training_output(config):
    # Making model weights directory
    training_dir = os.path.join(config.base_path, config.training_dir, config.name)
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    config.training_dir = training_dir

    # Making tensorboard directory
    tensorboard_dir = os.path.join(config.base_path, config.tensorboard_dir, config.name)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    config.tensorboard_dir =  tensorboard_dir
    
    yaml.safe_dump(edict2dict(config), open(training_dir + '/config.yml', 'w'))