# Customized PPO Training with Parameter Initialization from Previous Training

This customized version of `brax.training.agents.ppo.train` allows users to continue training their model by initializing parameters from a previous training session.
By providing the `previous_params` parameter, users can load the parameters from a previous run and start training from that point onwards, rather than starting with randomized parameters every time.
## Requirement
You need to either clone this repo or copy `brax.training.agents.ppo.train` and `brax.training.agents.ppo.networks` files and replace into your python library.

## Usage
After your first training `train_fn`, save your parameters. 

```python
train_fn = functools.partial(
    ppo.train, num_timesteps=30_000_000, ... YOUR OWN PARAMETERS)

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)
model.save_params(model_file_path, params)

```
To continue your training from the previous run, simply load the parameters and feed them into previous_params.

```python

previous_params = model.load_params(model_file_path)

train_fn = functools.partial(
    ppo.train, num_timesteps=30_000_000, ... , **previous_params = previous_params**)

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)
```
