import json

from exprimo import DeviceGraph, ComputationGraph, Simulator
from exprimo.optimizers import SimulatedAnnealingOptimizer, RandomHillClimbingOptimizer, GAOptimizer, \
    exponential_multiplicative_decay
from optimizers.utils import prefix_heuristic


def log(string, end='\n'):
    print(string, end=end)

    with open('../experiment_results/log.txt', 'a') as f:
        f.write(string + end)


experiments = [
    {
        'net_paths': ['../nets/mnist.json', '../nets/alex_v2.json', '../nets/resnet50.json'],
        'colocation_heuristic': None,
        'device_path': '../device_graphs/cluster2.json',
        'batches': 1,
        'pipeline_batches': 1,
        'function_evaluations': 25000
    },
    {
        'net_paths': ['../nets/mnist.json', '../nets/alex_v2.json', '../nets/resnet50.json'],
        'colocation_heuristic': None,
        'device_path': ['../device_graphs/cluster2-reduced-mnist.json', '../device_graphs/cluster2-reduced-alex.json',
                        '../device_graphs/cluster2-reduced-resnet.json'],
        'batches': 1,
        'pipeline_batches': 1,
        'function_evaluations': 25000
    },
    {
        'net_paths': ['../nets/mnist.json', '../nets/alex_v2.json', '../nets/resnet50.json'],
        'colocation_heuristic': None,
        'device_path': '../device_graphs/cluster2.json',
        'batches': 10,
        'pipeline_batches': 2,
        'function_evaluations': 25000
    },
]

repeats = 10

for e, experiment in enumerate(experiments):
    log(f'************  Experiment {e + 1}/{len(experiments)}  *************')
    batches = experiment['batches']
    pipeline_batches = experiment['pipeline_batches']
    evals = experiment['function_evaluations']
    pop_size = 50
    hc_optimizer = RandomHillClimbingOptimizer(steps=evals, batches=batches, pipeline_batches=pipeline_batches)
    sa_optimizer = SimulatedAnnealingOptimizer(steps=evals, temp_schedule=exponential_multiplicative_decay(50, 0.98),
                                               batches=batches, pipeline_batches=pipeline_batches)
    ga_optimizer = GAOptimizer(steps=evals/pop_size, elite_size=10, mutation_rate=0.05, use_caching=True,
                               batches=batches, pipeline_batches=pipeline_batches)
    optimizers = [hc_optimizer, sa_optimizer, ga_optimizer]

    for n, net_path in enumerate(experiment['net_paths']):
        log(f'Net {n+1}/{len(experiment["net_paths"])}: {net_path}')

        with open(net_path) as f:
            net_string = f.read()

        if type(experiment['device_path']) == list:
            device_path = experiment['device_path'][n]
        else:
            device_path = experiment['device_path']
        device_graph = DeviceGraph.load_from_file(device_path)

        log(f'{device_path}\n')

        colocation_heuristic = None
        if experiment['colocation_heuristic']:
            if type(experiment['colocation_heuristic']) == list:
                colocation_heuristic = experiment['colocation_heuristic'][n]
            else:
                colocation_heuristic = experiment['coloation_heuristic']

        for o, optimizer in enumerate(optimizers):
            log(f'Optimizer {o + 1}/{len(optimizers)}: {optimizer}')

            optimizer.colocation_heuristic = colocation_heuristic
            times = []

            for i in range(repeats):
                print(f'Repeat {i + 1}/{repeats}')
                best_net = optimizer.optimize(net_string, device_graph)

                graph = ComputationGraph()
                graph.load_from_string(best_net)
                simulator = Simulator(graph, device_graph)
                execution_time = simulator.simulate(batch_size=128, print_memory_usage=False, print_event_trace=False,
                                                    batches=batches, pipeline_batches=pipeline_batches)

                times.append(execution_time)
                print(f'{execution_time:.2}ms')
                with open(f'../experiment_results/nets/{e + 1}-{n + 1}-{o + 1}-{i + 1}.json', 'w') as f:
                    f.write(best_net)

            log(f'{",".join(times)}')
        log('\n')

    log('\n\n')

