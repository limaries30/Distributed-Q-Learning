from utils import make_file_dir, plot_and_save, count_bar_plot, save_data
import jax
from train import make_train

if __name__=="__main__":
    '''
    Run experiments for the Markov Congestion Game.
    Change configurations in learning rate
    '''
    LRS = [0.01, 0.02, 0.05]
    NUM_AGENTS = [8]
    for lr in LRS:
        for num_agents in NUM_AGENTS:

            config = {
                "EXP_ID" : "0c1e9",
                "SEED" : 8,
                "EPS_START" : 1,
                "EPS_END" : 0.2,
                "LR" : lr, 
                "GRAPH_TYPE" : 'ring',
                "NUM_UPDATES" : 640000,
                "NUM_AGENTS" : num_agents,
                "NUM_ACTIONS" : 4,
                "NUM_SEEDS" : 10, 
                "EPS_DECAY_TIME" : 80000,
                "GAMMA" : 0.9,
                "NUM_STATES" : 2,
                "ENV_KWARGS":{
                    "max_safe_capacity": 4,
                    "max_distance_capacity" :2
                }
            }
            rng = jax.random.PRNGKey(config["SEED"])
            rngs = jax.random.split(rng,config["NUM_SEEDS"])

            out =jax.vmap (jax.jit(make_train(config)))(rngs)
            print('reward',out['reward'][-10:])
            suffix = f'{config["GRAPH_TYPE"]}-{config["NUM_AGENTS"]}-{config["LR"]}'
            import matplotlib.pyplot as plt
            import os
            root_figures_dir = f'./figures/{config["EXP_ID"]}'
            root_results_dir = f'./results/{config["EXP_ID"]}'
            log_keys = ['regret','reward','epsilon','state', 'safe_evaluated_action', 'distance_evaluated_actions','consensus_error']
            for log_key in log_keys:
                make_file_dir(f'{root_figures_dir}/{log_key}')
                make_file_dir(f'{root_results_dir}/{log_key}')
                if log_key== 'safe_evaluted_action' or log_key=='distance_evaluated_actions':
                    save_data(root_results_dir=root_results_dir, data=out, data_name=log_key, batch_stat="count_mean", file_name=suffix, num_x_values=config['NUM_ACTIONS'])
                else:
                    save_data(root_results_dir=root_results_dir, data=out, data_name=log_key, batch_stat="mean", file_name=suffix)



            # plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="regret", file_name=suffix, batch_stat="mean", ylabel=r'$r_k-r^*$')
            # plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="reward", file_name=suffix, batch_stat="mean", ylabel=r'$r_k$')
            # plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="state", file_name=suffix, batch_stat="first", ylabel=r'$r_k-r^*$')
            # plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="epsilon", file_name=suffix, batch_stat="first", ylabel=r'$\epsilon$')
            # count_bar_plot(root_figures_dir=root_figures_dir, data = out, data_name="safe_evaluated_action", batch_stat="mean", file_name=suffix, num_x_values=config['NUM_ACTIONS'], ylabel=r'$r_k-r^*$')
            # count_bar_plot(root_figures_dir=root_figures_dir, data = out, data_name="distance_evaluated_actions", batch_stat="mean", file_name=suffix, num_x_values=config['NUM_ACTIONS'], ylabel=r'$r_k-r^*$')
            
            # plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="consensus_error", batch_stat="mean", file_name=suffix, yscale='log', ylabel='Consensus Error')
    
    
            
