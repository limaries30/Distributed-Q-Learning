This is the code for **Distributed Q-learning** (to appear in [RLC 2025](https://openreview.net/forum?id=NiPeCuZUb6)).

### About the Paper

- The paper studies finite-time bound of distributed Q-learning in a networked decentralized learning environment.

### What is this repo about?
- Experiments on distributed Q-learning in the **Markov congestion game** introduced by [Leonardos et al. (2021)](https://arxiv.org/abs/2106.01969).

### Details 
- The configs file (e.g., learning rate, number of agents, etc.) can be edited in the `run_multiple_exps.py` file.
- The code is based on jax==0.6.1.
- The figures are saved under `./figures/{exp_id}/`. The exp_id can be set by chaning the config params. 
- The utils for saving figures are in the file `./utils.py`


