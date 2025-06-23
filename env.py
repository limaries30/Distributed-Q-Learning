import jax.numpy as jnp
import jax

class MarokvCongestionGame:


    def __init__(self, num_agents, num_actions, max_safe_capacity, max_distance_capacity):

        '''
        A Markov Congestion Game environment.
        Args:
            num_agents (int): Number of agents in the game.
            num_actions (int): Number of actions available to each agent.
            max_safe_capacity (int): Maximum safe capacity for the agents. If the number of agents exceeds this capacity, the state transitions to a distance state.
        '''

        self.num_agents = num_agents
        self.num_actions = num_actions
        self.max_safe_capacity = max_safe_capacity
        self.max_distance_capacity = max_distance_capacity 
 
        self.SAFE_REWARD = jnp.array([[i+1 for i in range(self.num_actions)]])
        self.DISTANCE_PENALTY = 10
        self.DISTANCE_REWARD = self.SAFE_REWARD - self.DISTANCE_PENALTY
        self.REWARD = jnp.array([self.SAFE_REWARD, self.DISTANCE_REWARD])
        self.SAFE_STATE = 0
        self.DISTANCE_STATE = 1

    
        self.optimal_reward = self.get_optimal_reward()
    
    def step(self, state, action):

        nums = jnp.sum(jax.vmap(lambda s: s==action)(jnp.arange(self.num_actions)),axis=1) 
                
        next_s = jax.lax.cond(
            state[0] == self.SAFE_STATE,
            lambda _: jax.lax.cond(
                jnp.any(nums > self.max_safe_capacity),
                lambda __: self.DISTANCE_STATE,
                lambda __: self.SAFE_STATE,
                operand=None,
            ),
            lambda _: jax.lax.cond(
                jnp.all(nums <= self.max_distance_capacity),
                lambda __: self.SAFE_STATE,
                lambda __: self.DISTANCE_STATE,
                operand=None,
            ),
            operand=None
        )

        rewards = self.REWARD[next_s, :]*nums
        rewards = rewards[0][action] #jax.vmap(lambda x: REWARD[next_s,x])(state)
        
        return rewards, next_s
    

    def get_optimal_reward(self, ):
    
        agents = [max(min(self.num_agents-self.max_safe_capacity*i,self.max_safe_capacity),0) for i in range(self.num_actions)]
        agents.reverse()
        agents = jnp.array(agents)
        agent_rewards =  agents*self.SAFE_REWARD
        mean_reward = jnp.sum(agents*agent_rewards) / self.num_agents
        return mean_reward


if __name__=="__main__":
    num_agents = 6
    num_a = 3
    env = MarokvCongestionGame(num_agents= num_agents, num_actions=num_a, max_safe_capacity=4)
    print('optimal reward', env.get_optimal_reward())
    import numpy as np
    def decode_action(action, idx):
        '''int'''
        actions = jax.vmap(lambda x:(action%(num_a**(x+1)))//(num_a**x))(jnp.arange(num_agents))
        print(action,actions)
        return actions[idx]
    actions = np.random.randint(0,num_a**num_agents,size=(num_agents,))
    print(actions)
    best_response_actions = jax.vmap(lambda x, idx: decode_action(x,idx))(actions,jnp.arange(num_agents))
    print(best_response_actions)
    print(env.step([0],best_response_actions))
