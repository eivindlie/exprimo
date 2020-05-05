import json
import os

import time

from torchvision import transforms
import torchvision
import torch.utils.data
import torch
from tqdm import tqdm

from exprimo import log
from exprimo.benchmarking.utils import load_model_with_placement

BATCH_SIZE = 128


def train_single_batch(model, data, criterion, optimizer):
    output = model(data[0])
    loss = criterion(output, data[1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train_single_batch_inception(model, data, criterion, optimizer):
    output, aux_output = model(data[0])
    loss1 = criterion(output, data[1])
    loss2 = criterion(aux_output, data[1])
    loss = loss1 + 0.4 * loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def benchmark_with_placement(model_type, placement='cuda:0', batches=50, drop_batches=1, lr=0.01, verbose=False,
                             device_map=None, gpu_memory_limit=None, return_memory_overflow=False,
                             drop_last=True):
    if verbose:
        print('Starting benchmark...')

    if model_type.lower() in ['resnet', 'resnet50', 'resnet-50']:
        model_type = 'resnet50'
    elif model_type.lower() in ['inception', 'inception_v3', 'inceptionv3']:
        model_type = 'inception'
    elif model_type.lower() in ['alexnet', 'alex', 'alex_v2']:
        model_type = 'alexnet'

    model, criterion, optimizer, input_device, output_device = load_model_with_placement(model_type, placement, lr=lr,
                                                                                         device_map=device_map)

    model.train()
    batch_times = []

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if model_type in ['resnet50', 'alexnet']:
        dataset = torchvision.datasets.FakeData(transform=preprocess, size=500)
    elif model_type == 'inception':
        dataset = torchvision.datasets.FakeData(transform=preprocess, image_size=(3, 299, 299), size=500)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last
    )

    b = 0
    while b < batches + drop_batches:
        for data in train_loader:
            if verbose:
                print(f'Batch {b + 1}/{batches + drop_batches}', end='')

            memory_exceeded = False
            memory_overflow = 0

            if gpu_memory_limit:
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(device=torch.device(f'cuda:{i}'))

            try:
                torch.cuda.synchronize()
                data = data[0].to(input_device), data[1].to(output_device)

                start = time.time()
                if model_type == 'inception':
                    train_single_batch_inception(model, data, criterion, optimizer)
                else:
                    train_single_batch(model, data, criterion, optimizer)
                torch.cuda.synchronize()
                end = time.time()

                if gpu_memory_limit:
                    for i in range(torch.cuda.device_count()):
                        if isinstance(gpu_memory_limit, int):
                            max_memory_usage = torch.cuda.max_memory_allocated(torch.device(f'cuda:{i}'))
                            memory_exceeded = memory_exceeded or max_memory_usage > (gpu_memory_limit * 10**9)
                            memory_overflow += max(max_memory_usage / 10**9 - gpu_memory_limit, 0)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    memory_exceeded = True
                    memory_overflow = -1
                else:
                    raise e

            if not memory_exceeded:
                batch_times.append((end - start) * 1000)

            if verbose:
                if memory_exceeded:
                    log('Memory exceeded')
                else:
                    log(f' {batch_times[-1]}ms')

            if memory_exceeded:
                if return_memory_overflow:
                    return -1, memory_overflow
                return -1

            b += 1
            if b >= batches + drop_batches:
                break

    del model, criterion, optimizer, input_device, output_device

    if return_memory_overflow:
        return batch_times[drop_batches:], memory_overflow
    return batch_times[drop_batches:]


def benchmark_all_placements(placement_directory, results_file, model_type, generation_divisible_by=None, last_gen=None,
                             verbose=False, batches=50, drop_batches=0, device_map=None, gpu_memory_limit=None,
                             drop_last=True):
    with open(results_file, 'w') as f:
        f.write('')

    def generation_filter(file):
        if not file.endswith('.json') or not file.startswith('gen_'):
            return False

        generation = int(file.replace('gen_', '').replace('.json', ''))

        divisible_by = generation_divisible_by or 1

        if last_gen:
            return generation % divisible_by == 0 and generation <= last_gen

        return generation % divisible_by == 0

    dir_list = os.listdir(placement_directory)
    dir_list = list(filter(generation_filter, dir_list))

    for i, file in enumerate(tqdm(dir_list)):
        with open(os.path.join(placement_directory, file)) as f:
            placement = json.load(f)

        generation = int(file.replace('gen_', '').replace('.json', ''))

        if verbose:
            log(f'Benchmarking placement {i+1}/{len(dir_list)}: {file}... ', end='')

        batch_times = benchmark_with_placement(model_type, placement, batches=batches, drop_batches=drop_batches,
                                               device_map=device_map, gpu_memory_limit=gpu_memory_limit,
                                               drop_last=drop_last)

        with open(results_file, 'a') as f:
            f.write(f'{generation:04}, {",".join(map(lambda x: str(x), batch_times))}\n')

        if verbose:
            log(f'{sum(batch_times)/len(batch_times):.2f}ms')
