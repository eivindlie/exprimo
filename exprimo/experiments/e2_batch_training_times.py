import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from exprimo.benchmarking.benchmark import benchmark_all_placements
from exprimo.optimize import optimize_with_config
from exprimo.utils.convert_nets_to_placements import convert_to_placement
from exprimo import PLOT_STYLE

sns.set(style=PLOT_STYLE)


config_path = 'configs/experiments/e2_ga-malvik-resnet50.json'
model_type = 'resnet'
batches = 50
NORMALIZE_PLOT = True
PLOT_INDIVIDUAL = False

if __name__ == '__main__':
    # Force checkpointing to occur every 5th generation
    with open(config_path) as f:
        config = json.load(f)
    config['optimizer_args']['checkpoint_period'] = 5

    log_dir = os.path.expanduser(config['log_dir'])

    # Create a set of configurations by running an optimization process with checkpointing
    optimize_with_config(config=config)

    # Convert all configurations to placements for benchmarking
    convert_to_placement(os.path.join(log_dir, 'checkpoints'), os.path.join(log_dir, 'checkpoints', 'placements'))

    # Benchmark all the placements that were created above
    benchmark_all_placements(os.path.join(log_dir, 'checkpoints', 'placements'), os.path.join(log_dir, 'batch_times.csv'),
                             model_type, batches=batches, drop_batches=0)


    # Load and plot benchmark results
    def plot_times(data, title, output_file=None):
        plt.plot(data)
        plt.xlabel('Batch')
        if NORMALIZE_PLOT:
            plt.ylabel('Residual batch training time (fraction of mean)')
        else:
            plt.ylabel('Batch training time (ms)')
        plt.title(title)

        if output_file:
            plt.savefig(output_file, bbox_inches='tight')

        plt.show()


    batch_times = pd.read_csv(os.path.join(log_dir, 'batch_times.csv'), header=None, index_col=0)
    if NORMALIZE_PLOT:
        means = batch_times.mean(axis=1)
        batch_times = batch_times.sub(means, axis=0).divide(means, axis=0)

    if PLOT_INDIVIDUAL:
        for i in range(batch_times.shape[0]):
            generation = batch_times.index[0].replace('gen_', '').replace('.json', '')
            times = batch_times.iloc[i, 1:]

            output_file = os.path.join(log_dir, 'batch_training_time', f'gen_{generation}.pdf')

            plot_times(times, f'Generation {generation}', output_file)
    else:
        avg_batch_times = batch_times.mean(axis=0)
        plot_times(avg_batch_times, 'Average batch time residuals (with last batch in dataset)',
                   output_file=os.path.join(log_dir, 'batch_times.pdf'))
