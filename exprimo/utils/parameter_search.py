import json

import itertools
from tqdm import tqdm

from exprimo.optimize import optimize_with_config


def do_parameter_search(config_path, parameter_grid, repeats=10, verbose=False):
    with open(config_path) as f:
        config = json.load(f)

    config['plot_event_trace'] = False

    args_blueprint = config.get('optimizer_args', {})

    if 'benchmarking_generations' in args_blueprint:
        args_blueprint['benchmarking_generations'] = 0
    if 'benchmarking_steps' in args_blueprint:
        args_blueprint['benchmarking_steps'] = 0
    if 'benchmark_before_selection' in args_blueprint:
        args_blueprint['benchmark_before_selection'] = False
    args_blueprint['verbose'] = False

    grid_rows = []
    grid_size = 1
    for key, value in parameter_grid.items():
        grid_size *= len(value)
        row = []
        for v in value:
            row.append((key, v))
        grid_rows.append(row)

    best_time = None
    best_params = None
    for i, combination in tqdm(enumerate(itertools.product(*grid_rows)), total=grid_size):
        if verbose:
            print(f'Testing combination {i}: {combination}...')

        if 'generations' in args:
            original_pop = args['population_size']

        args = args_blueprint.copy()
        args = dict(tuple(args.items()) + combination)

        if 'generations' in args:
            args['generations'] = args['generations'] * (original_pop / args['population_size'])

        current_config = config.copy()
        current_config['optimizer_args'] = args

        best_times = [optimize_with_config(config=current_config, verbose=False)[1] for _ in tqdm(range(repeats))]

        mean_time = sum(best_times) / len(best_times)

        if verbose:
            print(f'{mean_time}ms')

        if best_time is None or mean_time < best_time:
            best_time = mean_time
            best_params = combination

    return dict(best_params)


if __name__ == '__main__':
    VERBOSE = True

    ga_grid = {
        'population_size': [30, 50, 100],
        'crossover_rate': [0.2, 0.4, 0.6, 0.8],
    }

    grid = (
        ga_grid,
    )[0]

    config_path = (
        'configs/ga-malvik-resnet50.json',
    )[0]

    best_params = do_parameter_search(config_path, grid, verbose=VERBOSE)
    print(f'Best parameters: \n{best_params}')
