import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from exprimo.benchmarking.benchmark import benchmark_all_placements
from exprimo import PLOT_STYLE, get_log_dir, optimize_with_config
from exprimo.utils.convert_nets_to_placements import convert_to_placement

sns.set(style=PLOT_STYLE)


config_path = 'configs/experiments/e5_ga-malvik-inception.json'
model_type = 'inception'

BATCHES = 20


def run_experiment():
    optimize_with_config(config_path, set_log_dir=True)

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


if __name__ == '__main__':
    run_experiment()

    plot_results(os.path.join(get_log_dir(), 'checkpoints', 'scores.csv'),
                 os.path.join(get_log_dir(), 'batch_times.csv'))
