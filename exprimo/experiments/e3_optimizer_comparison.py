import json
import multiprocessing
import os
from itertools import repeat, product

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from exprimo import set_log_dir, log, PLOT_STYLE
from exprimo.optimize import optimize_with_config

sns.set(style=PLOT_STYLE)

LOG_DIR = os.path.expanduser('~/logs/e3_optimizer-comparison')
set_log_dir(LOG_DIR)

run_config = (1, 0, 0, 1)
NETWORK = ('resnet50', 'alexnet', 'inception')[run_config[0] if isinstance(run_config[0], int) else 0]
BATCHES = (1, 10)[run_config[1] if isinstance(run_config[1], int) else 0]
PIPELINE_BATCHES = (1, 2, 4)[run_config[2] if isinstance(run_config[2], int) else 0]
MEMORY_LIMITED = bool(run_config[3] if len(run_config) > 3 and isinstance(run_config[3], int) else 0)

REPEATS = 50

OPTIMIZERS = ('hc', 'sa', 'ga', 'me')
OPTIMIZER_NAMES = {
    'hc': 'Hill Climbing',
    'sa': 'Simulated Annealing',
    'ga': 'Genetic Algorithm',
    'me': 'MAP-elites',
}


def test_optimizer(c, r, log_dir):
    c['log_dir'] = log_dir + f'/{r:03}'
    _, t = optimize_with_config(config=c, verbose=False, set_log_dir=True)
    return t


def run_optimizer_test(n_threads=-1):
    if n_threads == -1:
        n_threads = multiprocessing.cpu_count()

    for optimizer in tqdm(OPTIMIZERS):
        # log(f'Testing optimizer {optimizer}')
        run_name = f'e3_{optimizer}-{NETWORK}{"-pipeline" if PIPELINE_BATCHES > 1 else ""}' \
                   f'{"-limited" if MEMORY_LIMITED else ""}'
        config_path = f'configs/experiments/e3/{run_name}.json'
        score_path = os.path.join(LOG_DIR, f'{run_name}_scores.csv')

        with open(score_path, 'w') as f:
            f.write('run, time\n')

        with open(config_path) as f:
            config = json.load(f)

        config['optimizer_args']['verbose'] = False
        config['optimizer_args']['batches'] = BATCHES
        config['optimizer_args']['pipeline_batches'] = PIPELINE_BATCHES
        log_dir = config['log_dir']

        threaded_optimizer = config['optimizer'] in ('ga', 'genetic_algorithm', 'map-elites', 'map_elites')

        if n_threads == 1 or threaded_optimizer:
            for r in tqdm(range(REPEATS)):
                time = test_optimizer(config, r, log_dir)
                with open(score_path, 'a') as f:
                    f.write(f'{r},{time}\n')
        else:
            worker_pool = multiprocessing.Pool(n_threads)
            times = worker_pool.starmap(test_optimizer, zip(repeat(config), (r for r in range(REPEATS)),
                                                            repeat(log_dir)))
            with open(score_path, 'a') as f:
                for r, t in enumerate(times):
                    f.write(f'{r},{t}\n')

        set_log_dir(LOG_DIR)


def plot_results():
    all_results = pd.DataFrame()
    # CREATE PLOT OF RESULTS
    for optimizer in OPTIMIZERS:
        run_name = f'e3_{optimizer}-{NETWORK}{"-pipeline" if PIPELINE_BATCHES > 1 else ""}' \
                   f'{"-limited" if MEMORY_LIMITED else ""}'
        score_path = os.path.join(LOG_DIR, f'{run_name}_scores.csv')
        scores = pd.read_csv(score_path, index_col=0, squeeze=True)
        all_results[OPTIMIZER_NAMES[optimizer]] = scores

    plt.figure(figsize=(8, 8))
    chart = sns.barplot(data=all_results)
    chart.set_xticklabels(
        chart.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-large'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'score_comparison.pdf'))
    plt.show()
    plt.close()


def plot_result_all_networks(test_type='normal'):
    all_results = pd.DataFrame()
    # CREATE PLOT OF RESULTS
    for network in ('alexnet', 'resnet50', 'inception'):
        for optimizer in OPTIMIZERS:
            run_name = f'e3_{optimizer}-{network}{"-pipeline" if test_type == "pipelined" else ""}' \
                       f'{"-limited" if test_type == "limited" else ""}'
            score_path = os.path.join(LOG_DIR, f'{run_name}_scores.csv')
            scores = pd.read_csv(score_path, index_col=0, squeeze=True)
            all_results['score'] = scores
            all_results['optimizer'] = OPTIMIZER_NAMES[optimizer]
            all_results['network'] = network

        chart = sns.barplot(x='network', y='score', hue='optimizer', data=all_results)
        chart.set_xticklabels(
            chart.get_xticklabels(),
            rotation=45,
            horizontalalignment='right',
            fontweight='light',
            fontsize='x-large'
        )
        plt.savefig(os.path.join(LOG_DIR, f'score_comparison_{test_type}.pdf'), bb_inches='tight')
        plt.show()
        plt.close()


def run_all_variants():
    networks = 'resnet50', 'alexnet', 'inception'
    test_types = 'normal', 'limited', 'pipelined'
    global BATCHES, PIPELINE_BATCHES, MEMORY_LIMITED, NETWORK
    for variation in tqdm(product(networks, test_types)):
        log(f'Testing f{variation[0]} network in {variation[1]} configuration')
        NETWORK = variation[0]
        if variation[1] == 'normal':
            BATCHES = 1
            PIPELINE_BATCHES = 1
            MEMORY_LIMITED = False
            REPEATS = 50
        elif variation[1] == 'limited':
            BATCHES = 1
            PIPELINE_BATCHES = 1
            MEMORY_LIMITED = True
            REPEATS = 50
        elif variation[1] == 'pipelined':
            BATCHES = 10
            PIPELINE_BATCHES = 4
            MEMORY_LIMITED = False
            REPEATS = 10

        run_optimizer_test()


if __name__ == '__main__':
    if run_config == 'all':
        run_all_variants()
    else:
        run_optimizer_test()
        plot_results()
