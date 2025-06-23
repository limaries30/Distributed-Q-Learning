import os
import matplotlib.pyplot as plt
import jax.numpy as jnp
from collections import Counter
import numpy as np


def make_file_dir(name:str):
    '''
    Create a directory if it does not exist. 
    '''
    if not os.path.exists(f'{name}'):
        os.makedirs(f'{name}')

def save_data(root_results_dir:str, data:np.array, data_name:str,
               batch_stat:str, file_name:str, num_x_values=None):
    '''
        Save data to a numpy file based on the specified batch statistic.
        Parameters:
        - batch_stat: Statistic to use for plotting ('mean', 'first').
    '''
    if batch_stat=="mean":
        jnp.save(f'{root_results_dir}/{data_name}/{file_name}.npy', jnp.mean(data[f'{data_name}'],axis=0))
    if batch_stat=="first":
        jnp.save(f'{root_results_dir}/{data_name}/{file_name}.npy', data[f'{data_name}'][0,:])
    if batch_stat=="count_mean":
        action_space = np.arange(num_x_values)
        counts = Counter(data[f'{data_name}'].reshape(-1).tolist())
        bar_values = np.array([counts.get(a, 0) for a in action_space])/len(data[f'{data_name}'])
        np.save(f'{root_results_dir}/{data_name}/{file_name}.npy', bar_values)


def plot_and_save(root_figures_dir, data, data_name, batch_stat, file_name, yscale=None, ylabel=None):
    '''
     Plot data and save the figure.
     Parameters:
     - root_figures_dir: Root directory to save the figure.
     - batch_stat: Statistic to use for plotting ('mean', 'first').
    '''
    plt.figure()
    plt.tight_layout()
    if batch_stat=="mean":
        plt.plot(jnp.mean(data[f'{data_name}'],axis=0)[100:])
    if batch_stat=="first":
        plt.plot(data[f'{data_name}'][0,:])
    if ylabel is not None:
        plt.ylabel(ylabel)
    if yscale is not None:
        plt.yscale(yscale)
    plt.savefig(f'{root_figures_dir}/{data_name}/{file_name}.png')
    
def count_bar_plot(root_figures_dir:str, data:np.array, data_name:str,
                    batch_stat:str, file_name:str, num_x_values:int, ylabel=None):
    
    ''' Plot a bar chart of action counts or means.
        Parameters:
        - root_figures_dir: Root directory to save the figure.
        - batch_stat: Statistic to use for plotting ('mean', 'first').
        - num_x_values: Number of unique actions to plot. (xticks)
    '''
    
    action_space = np.arange(num_x_values)
    x = np.arange(len(action_space)) 


    plt.figure()
    plt.tight_layout()
    

    if batch_stat == "first":
        num_gropus = 3
        width = 0.8 /num_gropus
        for i, actions in enumerate(data[f'{data_name}'][-num_gropus:]):
            counts = Counter(actions.tolist())
            bar_values = [counts.get(a, 0) for a in action_space]
            plt.bar(x + i * width, bar_values, width=width, label=i)
        plt.xticks(x + width, action_space)
        plt.savefig(f'{root_figures_dir}/{data_name}/{file_name}.png')
    if batch_stat == "mean":

        counts = Counter(data[f'{data_name}'].reshape(-1).tolist())

        bar_values = np.array([counts.get(a, 0) for a in action_space])/len(data[f'{data_name}'])
        plt.bar(x , bar_values)
        plt.xticks(action_space)
        plt.savefig(f'{root_figures_dir}/{data_name}/{file_name}.png')
