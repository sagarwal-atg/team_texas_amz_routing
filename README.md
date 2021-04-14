# team_texas_amz_routing

## Recommended file structure
> amazon (main folder):
>      -- data (all JSON files)
>      -- runs (tensorboard)
>      -- trained_models (saving trained model weights)
>      -- team_texas_amz_routing (this GitHub repo)


## Setting up the environment
`pip3 install -r requirments.txt`

Add new libraries in requirments.txt.

Add "virtualenv" instructions (TODO).


## Training the model

`python3 team_texas_amz_routing/irl_train.py linear_irl`


## Eval the model

`python3 team_texas_amz_routing/eval.py linear_irl/model_900.pt`

