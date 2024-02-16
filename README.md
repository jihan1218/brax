# Customized PPO Training with Parameter Initialization from Previous Training

This customized version of `brax.training.agent.ppo.train` allows users to continue training their model by initializing parameters from a previous training session.
By providing the `previous_params` parameter, users can load the parameters from a previous run and start training from that point onwards, rather than starting with randomized parameters every time.

## Usage
After your first training `train_fn`, save your parameters. 

```python

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)
model.save_params(model_path, params)

```
To continue your training from the previous run, simply load the parameters and feed them into previous_params.

```python

previous_params = model.load_params(model_path)

train_fn = functools.partial(
    ppo.train, num_timesteps=30_000_000, ... , previous_params = previous_params)

```
