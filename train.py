'''
A markov congestiong game model based on the paper 'Global Convergence of Multi-Agent Policy Gradient in
Markov Potential Games'.
'''

import numpy as np
import jax
import jax.numpy as jnp
from graph_utils import generate_mixing_matrix, get_graph
from utils import make_file_dir, plot_and_save, count_bar_plot
from env import MarokvCongestionGame



def make_train(config):
    num_agents = config["NUM_AGENTS"]
    graph_type = config["GRAPH_TYPE"]
    num_updates = config["NUM_UPDATES"]

    num_a = config["NUM_ACTIONS"]
    num_s = config["NUM_STATES"]


    def train(rng):
        
        joint_a = num_a ** num_agents
        
        env = MarokvCongestionGame(num_agents=num_agents, num_actions=num_a, **config["ENV_KWARGS"])

        #rng = jax.random.PRNGKey(config["SEED"])
        rng, _rng = jax.random.split(rng)
        
        # Initialize parameters which is defind over joitn action space
        #params = jnp.full((num_agents, num_s, joint_a), 1.0)
        params = jax.random.normal(key=rng,shape=(num_agents, num_s, joint_a))

        epsilon_start = config["EPS_START"]
        epsilon_end = config["EPS_END"]
        eps_decay_time = config["EPS_DECAY_TIME"]

        gamma = config["GAMMA"]
        
        
        lr = config["LR"]
        graph = get_graph(graph_type, num_agents)
        W = generate_mixing_matrix(graph)
        num_steps = 0
        def greedy_action(params, rng):
            def sample_greedy(p, rng):
                max_val = jnp.max(p)
                is_max = p == max_val
                # Assign high score to max elements, then add tiny random noise to break ties
                noise = jax.random.uniform(rng, shape=p.shape) * 1e-6
                scores = is_max.astype(jnp.float32) + noise
                return jnp.argmax(scores)

            rngs = jax.random.split(rng, params.shape[0])
            return jax.vmap(sample_greedy)(params, rngs)

            
        def decode_action(action:int, idx):
            '''Given a joint action, decode the action for agent idx.'''
            actions = jax.vmap(lambda x:(action%(num_a**(x+1)))//(num_a**x))(jnp.arange(num_agents))
            return actions[idx]


        def _update(runner_state, unused):
            params, state, num_steps, rng = runner_state
                    

            def random_action(rng):
                action = jax.random.randint(rng, minval=0, maxval=joint_a, shape=(num_agents,))
                return action
            
            def softmax_action(params, rng):
                '''
                params : (num_s, join_a)
                '''
                rng, _rng = jax.random.split(rng)
                exp_params = jnp.exp(3*params)
                probs = exp_params / jnp.sum(exp_params, axis=-1, keepdims=True)
                action = jax.random.categorical(_rng, jnp.log(probs))
                #jax.debug.print('softmax action={action}', action=action.shape)
                return action.squeeze()

            def eps_greedy(params, epsilon, rng):
                '''
                    params : (num_s, join_a)
                '''
                rng, _rng = jax.random.split(rng)
                rand_val = jax.random.uniform(_rng, minval=0, maxval=1)
                rng, _rng = jax.random.split(rng)
                action = jax.lax.cond(rand_val<epsilon, lambda _: random_action(_rng), \
                                      lambda _: greedy_action(params,rng), operand=None)  # (num_agents,)
                

                return action


            rng, _rng = jax.random.split(rng)    
            epsilon = epsilon_end + jnp.exp(-(epsilon_start-epsilon_end)*num_steps/eps_decay_time)
            num_steps+=1
            actions = softmax_action(params[:,state,:], _rng)  
            # actions = eps_greedy(params[:,state,:], epsilon, _rng)
            #jax.debug.print('eps greedy actions={actions}', actions=actions.shape)
            best_response_actions = jax.vmap(lambda x, idx: decode_action(x,idx))(actions,jnp.arange(num_agents))
            #one_hot_encoded_action = jax.nn.one_hot(best_response_actions,num_a)   
            
            rewards, next_state = env.step(state, best_response_actions) #get_reward(best_response_actions,num_a)
                
            Wq = W@(params.reshape(num_agents,-1))
                       
            
            
            powers = jnp.array([num_a**i for i in range(num_agents)])
            action_idx = jnp.sum(powers@best_response_actions).astype(jnp.int32) 
            q_vals=params[jnp.arange(num_agents),state,action_idx]
            
            next_q_vals = jnp.max(params[:,next_state,:],axis=-1)
            step_avg_reward = jnp.mean(rewards)

            next_params = Wq #+ lr*jax.vmap(lambda s:s*phi)(rewards+gamma*next_q_vals-q_vals)
            next_params = next_params.reshape(num_agents, num_s, joint_a)
            #jax.debug.print('shape1={shape1}', shape1=next_params[jnp.arange(num_agents),state,best_response_actions].shape)
            next_params = next_params.at[jnp.arange(num_agents),state,action_idx].add(lr*(rewards+gamma*next_q_vals-q_vals))
            #jax.debug.print('shape2={shape2}', shape2=(rewards+gamma*next_q_vals-q_vals).shape)
            
            avg = jnp.mean(next_params,axis=0)
            consensus_error = jnp.linalg.norm(next_params - avg)

            runner_state = (next_params, jnp.array([next_state]), num_steps, rng)
            regret = step_avg_reward-env.optimal_reward
            metric = {'reward': step_avg_reward, "action":best_response_actions,\
                       "epsilon":epsilon, "state": state, "regret": regret, "consensus_error":consensus_error }
            
            return runner_state, metric
        state = jax.random.randint(rng,minval=0,maxval=1,shape=(1,))
        runner_state = (params, state, num_steps, rng)


        def evaluate(params, state, rng):
            '''
            Evaluate the current policy parameters.
            Args:
                params: (num_agents, num_s, joint_a)
                state: (num_agents,)
                rng: jax random key
            Returns:
                action: (num_agents,)
                reward: (num_agents,)
            '''
            rng, _rng = jax.random.split(rng)
            actions = greedy_action(params[:, state, :], _rng)
            return actions
        

        runner_state, metric = jax.lax.scan(_update, runner_state, None, length =num_updates)
        final_params, state, num_steps, rng = runner_state
        safe_evaluated_actions = evaluate(final_params, 0, rng) # evaluate the policy at safe state after training
        safe_evaluated_best_response_actions = jax.vmap(lambda x, idx: decode_action(x,idx))(safe_evaluated_actions,jnp.arange(num_agents))
        metric["safe_evaluted_action"] = safe_evaluated_best_response_actions


        distance_evaluated_actions = evaluate(final_params, 1, rng) # evaluate the policy at safe state after training
        distance_evaluated_best_response_actions = jax.vmap(lambda x, idx: decode_action(x,idx))(distance_evaluated_actions,jnp.arange(num_agents))
        metric["distance_evaluated_actions"] = distance_evaluated_best_response_actions


        return metric
    return train


if __name__=="__main__":

    config = {
        "SEED" : 8,
        "EPS_START" : 1,
        "EPS_END" : 0.2,
        "LR" : 0.01, 
        "GRAPH_TYPE" : 'ring',
        "NUM_UPDATES" : 100,
        "NUM_AGENTS" : 8,
        "NUM_ACTIONS" : 4,
        "NUM_SEEDS" : 10, 
        "EPS_DECAY_TIME" : 10,
        "GAMMA" : 0.99,
        "EXP_ID" : "1",
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
    log_keys = ['regret','reward','epsilon','state', 'safe_evaluted_action', 'distance_evaluated_actions','consensus_error']
    for log_key in log_keys:
        make_file_dir(f'{root_figures_dir}/{log_key}')


    plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="regret", file_name=suffix, batch_stat="mean", ylabel=r'$r_k-r^*$')
    plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="reward", file_name=suffix, batch_stat="mean", ylabel=r'$r_k$')
    plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="state", file_name=suffix, batch_stat="first", ylabel=r'$r_k-r^*$')
    plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="epsilon", file_name=suffix, batch_stat="first", ylabel=r'$\epsilon$')
    count_bar_plot(root_figures_dir=root_figures_dir, data = out, data_name="safe_evaluted_action", batch_stat="mean", file_name=suffix, num_x_values=config['NUM_ACTIONS'], ylabel=r'$r_k-r^*$')
    count_bar_plot(root_figures_dir=root_figures_dir, data = out, data_name="distance_evaluated_actions", batch_stat="mean", file_name=suffix, num_x_values=config['NUM_ACTIONS'], ylabel=r'$r_k-r^*$')
    
    plot_and_save(root_figures_dir=root_figures_dir, data = out, data_name="consensus_error", batch_stat="mean", file_name=suffix, yscale='log', ylabel='Consensus Error')

