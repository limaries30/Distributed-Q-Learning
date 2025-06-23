This is the code for **Distributed Q-learning** (to appear in [RLC 2025](https://openreview.net/forum?id=NiPeCuZUb6)).


### Details
- Experiments for the **Markov congestion game** introduced in [Leonardos et al. (2021)](https://arxiv.org/abs/2106.01969).

- The configs file (e.g., learning rate, number of agents, etc.) can be edited in the `run_multiple_exps.py` file.

### Plotting Figures
- The figures are saved under `./figures/{exp_id}/`. The exp_id can be set by chaning the config params. 
- The utils for saving figures are in the file `./utils.py`
