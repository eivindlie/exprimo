import json
import os

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from exprimo import set_log_dir, log, PLOT_STYLE
from exprimo.optimize import optimize_with_config

sns.set(style=PLOT_STYLE)

LOG_DIR = os.path.expanduser('~/logs/e3_optimizer-comparison')
set_log_dir(LOG_DIR)

run_config = (0, 0, 0)
NETWORK = ('resnet50', 'alexnet', 'inception')[run_config[0]]
BATCHES = (1, 10)[run_config[1]]
PIPELINE_BATCHES = (1, 2, 4)[run_config[2]]
MEMORY_LIMITED = False

REPEATS = 50

OPTIMIZERS = ('hc', 'sa', 'ga', 'me')
OPTIMIZER_NAMES = {
    'hc': 'Hill Climbing',
    'sa': 'Simulated Annealing',
    'ga': 'Genetic Algorithm',
    'me': 'MAP-elites',
}


def run_optimizer_test():
    for optimizer in tqdm(OPTIMIZERS):
        # log(f'Testing optimizer {optimizer}')
        run_name = f'e3_{optimizer}-{NETWORK}{"-pipeline" if PIPELINE_BATCHES > 1 else ""}'
        config_path = f'configs/experiments/{run_name}.json'
        score_path = os.path.join(LOG_DIR, f'{run_name}{"-limited" if MEMORY_LIMITED else ""}_scores.csv')

        with open(score_path, 'w') as f:
            f.write('run, time\n')

        with open(config_path) as f:
            config = json.load(f)

        config['optimizer_args']['verbose'] = False

        for r in tqdm(range(REPEATS)):
            _, time = optimize_with_config(config=config, verbose=False, set_log_dir=True)
            with open(score_path, 'a') as f:
                f.write(f'{r},{time}\n')
        set_log_dir(LOG_DIR)


def plot_results():
    all_results = pd.DataFrame()
    # CREATE PLOT OF RESULTS
    for optimizer in OPTIMIZERS:
        run_name = f'e3_{optimizer}-{NETWORK}{"-pipeline" if PIPELINE_BATCHES > 1 else ""}'
        score_path = os.path.join(LOG_DIR, f'{run_name}{"-limited" if MEMORY_LIMITED else ""}_scores.csv')
        scores = pd.read_csv(score_path, index_col=0, squeeze=True)
        all_results[OPTIMIZER_NAMES[optimizer]] = scores

    sns.barplot(data=all_results)
    plt.savefig(os.path.join(LOG_DIR, 'score_comparison.pdf'))
    plt.show()


if __name__ == '__main__':
    run_optimizer_test()
    plot_results()
