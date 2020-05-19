import json
import os

import numpy as np
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from exprimo.benchmarking.benchmark import benchmark_all_placements
from exprimo import PLOT_STYLE, get_log_dir, optimize_with_config, set_log_dir
from exprimo.utils.convert_nets_to_placements import convert_to_placement

sns.set(style=PLOT_STYLE)


model_type = 'inception'
config_path = f'configs/experiments/e5_ga-malvik-{model_type}.json'
repeats = 10
with open(config_path) as f:
    log_dir = json.load(f)['log_dir']

BATCHES = 20


def run_experiment(lg_dir):
    set_log_dir(lg_dir)
    optimize_with_config(config_path)

    convert_to_placement(os.path.join(get_log_dir(), 'checkpoints'),
                         os.path.join(get_log_dir(), 'checkpoints', 'placements'))

    benchmark_all_placements(os.path.join(get_log_dir(), 'checkpoints', 'placements'),
                             os.path.join(get_log_dir(), 'batch_times.csv'),
                             model_type, batches=BATCHES, drop_batches=1, format='long')


def plot_results(sim_path, real_path):
    sim_results = pd.read_csv(sim_path, names=['generation', 'time'])
    sim_results['category'] = 'Simulated'
    real_results = pd.read_csv(real_path, names=['generation', 'time'])
    real_results['category'] = 'Benchmarked'

    all_results = pd.concat([sim_results, real_results], axis=0)

    cmap = sns.cubehelix_palette(2, start=.5, rot=-.75, light=0.5, reverse=True)
    sns.lineplot(x='generation', y='time', hue='category', style='category', data=all_results, palette=cmap)
    plt.legend(['Simulated', 'Benchmarked'])
    plt.xlabel('Generation')
    plt.ylabel('Batch execution time (ms)')

    plt.tight_layout()
    plt.savefig(os.path.join(get_log_dir(), 'sim_real_comp.pdf'))

    plt.show()
    plt.close()


def run_n_times(n):
    for i in tqdm(range(n)):
        run_experiment(lg_dir=os.path.join(log_dir, f'{i:03}'))

        plot_results(os.path.join(get_log_dir(), 'checkpoints', 'scores.csv'),
                     os.path.join(get_log_dir(), 'batch_times.csv'))


def scatter_plot_all_runs(lg_dir, use_benchmark_mean=True, plot_regression=True):
    all_results = pd.DataFrame()

    for path in os.listdir(lg_dir):
        if not re.search('^[0-9]+$', path):
            continue

        real_scores = pd.read_csv(os.path.join(lg_dir, path, 'batch_times.csv'), index_col=0,
                                  names=['generation', 'time'])
        real_scores.index = real_scores.index.astype(int)

        if use_benchmark_mean:
            real_scores = real_scores.groupby(real_scores.index).mean()

        sim_scores = pd.read_csv(os.path.join(lg_dir, path, 'checkpoints', 'scores.csv'), index_col=0,
                                 names=['generation', 'time'])
        sim_scores.index = sim_scores.index.astype(int)

        combined_scores = real_scores.join(sim_scores, lsuffix='_benchmarked', rsuffix='_simulated')

        all_results = pd.concat([all_results, combined_scores])

    x, y = all_results['time_simulated'], all_results['time_benchmarked']
    plt.scatter(x, y)

    if plot_regression:
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * x + b, c='orange', ls='--')

        corr = np.corrcoef(x, y)
        print(f'Pearson coefficient: R = {corr[0][1]}')

    plt.xlabel('Simulated batch time (ms)')
    plt.ylabel('Benchmarked batch time (ms)')

    plt.tight_layout()

    plt.savefig(os.path.join(lg_dir, 'scatter_plot.pdf'))
    plt.show()


if __name__ == '__main__':
    if repeats == 1:
        run_experiment(log_dir=log_dir)
        plot_results(os.path.join(get_log_dir(), 'checkpoints', 'scores.csv'),
                     os.path.join(get_log_dir(), 'batch_times.csv'))
    else:
        run_n_times(repeats)
