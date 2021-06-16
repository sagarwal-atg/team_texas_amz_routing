# team_texas_amz_routing

## Recommended file structure
> amazon (main folder):  \
>      -- data (all JSON files) \
>      -- runs (tensorboard) \
>      -- trained_models (saving trained model weights) \
>      -- team_texas_amz_routing (this GitHub repo) \


## Setting up the environment
`pip3 install -r requirments.txt`

Add new libraries in requirments.txt.

Add "virtualenv" instructions (TODO).


## Training the model

`python3 train_irl_nn.py --config_path configs/irl_config.yaml`


## Eval the model

`python3 eval_irl_nn.py --config_path configs/eval_irl_nn_config.yaml`

## Notes
- small_data folder is a subset of the first 20 routes from the data/model_build_inputs
