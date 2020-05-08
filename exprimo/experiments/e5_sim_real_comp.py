import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from exprimo import PLOT_STYLE, get_log_dir

sns.set(style=PLOT_STYLE)


def plot_results(sim_path, real_path):
    sim_results = pd.read_csv(sim_path)
    sim_results['category'] = 'Simulated'
    real_results = pd.read_csv(real_path)
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
    plot_results(os.path.expanduser('~/logs/e5-2/time_history.csv'),
                 os.path.expanduser('~/logs/e5-2/batch_times_long2.csv'))
