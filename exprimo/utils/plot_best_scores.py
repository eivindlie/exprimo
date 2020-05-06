import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from exprimo import PLOT_STYLE

sns.set(style=PLOT_STYLE)

results_file = '~/logs/e4_solution_transfer/2020-04-07_malvik_resnet_200sim_100real_limited-memory.csv'
plot_file = '~/logs/e4_solution_transfer/2020-04-07_malvik_resnet_200sim_100real_limited-memory.svg'
plot_title = f'ResNet on Malvik - 200 simulated, 100 benchmarked generations\nLimited memory'

benchmark_limit = 200

if __name__ == '__main__':
    data = pd.read_csv(os.path.expanduser(results_file), index_col=0, skiprows=0, names=['generation', 'batch_time'])

    data.plot(legend=None)

    if benchmark_limit:
        plt.axvline(x=benchmark_limit, c='grey', ls='--')

    plt.title(plot_title)
    plt.xlabel('Generation')
    plt.ylabel('Batch training time (ms)')

    plt.savefig(os.path.expanduser(plot_file), bb_inches='tight')

    plt.show()
    plt.close()

